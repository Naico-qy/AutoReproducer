"""Evaluation entrypoint.

Computes the two evaluation metrics described in §4.1 of the paper + addendum:

1. **Average reward** of a policy over ``n_eval_episodes`` episodes — used
   for Tables 1 and Figure 2 (refining performance).

2. **Fidelity score** of an explanation method —
   ``log(d/d_max) − log(l/L)``  where ``d`` is the absolute reward change
   when randomising actions in the top-K window and ``d_max`` is the
   environment's per-episode reward bound.

The script writes a JSON summary to ``--output`` (default ``/output/metrics.json``)
that the PaperBench reproduction grader can ingest directly.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
import yaml

from data import make_env
from model import ActorCritic, MaskNet, fidelity_score
from model.explanation import _rollout_with_scores  # internal helper


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate_reward(
    env, policy: ActorCritic, n_episodes: int = 100, device: str = "cpu"
) -> Dict[str, float]:
    """Mean episode reward (paper Table 1 / Figure 2 metric)."""
    rewards: List[float] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_r = 0.0
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(
                0
            )
            with torch.no_grad():
                action, _, _ = policy.act(obs_t, deterministic=True)
            obs, r, term, trunc, _ = env.step(action.squeeze(0).cpu().numpy())
            ep_r += float(r)
            done = term or trunc
        rewards.append(ep_r)
    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "n": int(n_episodes),
    }


def evaluate_fidelity(
    env,
    policy: ActorCritic,
    mask_net: MaskNet,
    Ks: List[float],
    n_traj: int,
    d_max: float,
    max_episode_steps: int,
    device: str = "cpu",
) -> Dict[str, dict]:
    """Per-K fidelity scores following addendum's pipeline."""
    out = {}
    for K in Ks:
        out[f"K={K}"] = fidelity_score(
            env,
            policy,
            mask_net,
            K=K,
            n_trajectories=n_traj,
            d_max=d_max,
            max_steps=max_episode_steps,
            device=device,
        )
    return out


def build_actor_critic(env, hidden=(64, 64)) -> ActorCritic:
    obs_dim = int(np.prod(env.observation_space.shape))
    if hasattr(env.action_space, "shape") and len(env.action_space.shape) > 0:
        action_dim = int(env.action_space.shape[0])
        continuous = True
    else:
        action_dim = int(env.action_space.n)
        continuous = False
    return ActorCritic(obs_dim, action_dim, hidden=hidden, continuous=continuous)


def maybe_load(model: torch.nn.Module, path: str) -> bool:
    if os.path.exists(path):
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="RICE evaluation script")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument(
        "--method",
        type=str,
        default="rice",
        help="Which refined-policy checkpoint to evaluate.",
    )
    parser.add_argument("--output", type=str, default="/output/metrics.json")
    parser.add_argument(
        "--skip_fidelity",
        action="store_true",
        help="Skip the fidelity-score computation.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    env_name = args.env or cfg["env"]["name"]
    env = make_env(
        env_name,
        seed=cfg["logging"]["seed"],
        max_episode_steps=cfg["env"]["max_episode_steps"],
    )
    device = "cpu"

    metrics: Dict[str, object] = {"env": env_name, "method": args.method}

    # --- Reward of the refined policy --------------------------------------
    refined = build_actor_critic(env)
    ref_path = os.path.join(args.checkpoint_dir, f"refine_{args.method}_{env_name}.pt")
    if maybe_load(refined, ref_path):
        metrics["refined_reward"] = evaluate_reward(
            env, refined, cfg["eval"]["n_eval_episodes"], device
        )
    else:
        print(f"[eval] no refined checkpoint at {ref_path}")
        metrics["refined_reward"] = None

    # --- Reward of the pre-trained ("No Refine") policy --------------------
    pre = build_actor_critic(env)
    pre_path = os.path.join(args.checkpoint_dir, f"pretrain_{env_name}.pt")
    if maybe_load(pre, pre_path):
        metrics["pretrain_reward"] = evaluate_reward(
            env, pre, cfg["eval"]["n_eval_episodes"], device
        )
    else:
        metrics["pretrain_reward"] = None

    # --- Fidelity of the mask network --------------------------------------
    if not args.skip_fidelity:
        obs_dim = int(np.prod(env.observation_space.shape))
        mask_net = MaskNet(obs_dim)
        mask_path = os.path.join(args.checkpoint_dir, f"mask_{env_name}.pt")
        if maybe_load(mask_net, mask_path) and metrics["pretrain_reward"] is not None:
            d_max = cfg["eval"]["d_max"].get(env_name, 1000.0)
            metrics["fidelity"] = evaluate_fidelity(
                env,
                pre,
                mask_net,
                Ks=cfg["eval"]["fidelity_top_K"],
                n_traj=cfg["eval"]["fidelity_n_trajectories"],
                d_max=d_max,
                max_episode_steps=cfg["env"]["max_episode_steps"],
                device=device,
            )
        else:
            metrics["fidelity"] = None

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[eval] wrote metrics to {args.output}")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
