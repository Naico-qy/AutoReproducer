"""Continual RL trainer for CompoNet (and baselines).

Usage:
    python train.py --config configs/default.yaml \
                    --sequence metaworld \
                    --method componet \
                    --seed 0 \
                    --budget-per-task 1000000 \
                    --output-dir /output

Implements the high-level loop of Section 5.2:
  for k in 1..N:
      reset critic / replay buffer at task boundary
      add new CompoNet module (freeze previous one)
      for t in 1..Delta:
          interact with environment
          update agent (SAC for Meta-World, PPO for ALE)
      evaluate success rate

A SHORT smoke run (--smoke) is provided for the PaperBench reproduction
container, in case 1M timesteps x 20 tasks is infeasible inside 24h.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from utils.common import build_actor, get_device, load_config, set_seed
from data.loader import (
    FREEWAY_SUCCESS_SCORES,
    SPACEINVADERS_SUCCESS_SCORES,
    METAWORLD_CW20_TASKS,
    SPACEINVADERS_MODES,
    FREEWAY_MODES,
    make_task_sequence,
    ReplayBuffer,
)
from algorithms.sac import SAC, SACConfig
from algorithms.ppo import PPO, PPOConfig
from algorithms.metrics import (
    average_performance,
    forgetting,
    success_rate_curve,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--sequence", type=str, default=None)
    p.add_argument("--method", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--budget-per-task", type=int, default=None, dest="budget_per_task")
    p.add_argument("--output-dir", type=str, default=None, dest="output_dir")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny smoke version (small Delta, 2 tasks).",
    )
    return p.parse_args()


def merge_cli(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    for key in (
        "sequence",
        "method",
        "seed",
        "budget_per_task",
        "output_dir",
        "device",
    ):
        v = getattr(args, key)
        if v is not None:
            cfg[key] = v
    return cfg


# ---------------------------------------------------------------------------
# Meta-World (SAC) trainer
# ---------------------------------------------------------------------------
def train_metaworld(cfg: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Run the Meta-World CW20 sequence with SAC."""
    sac_cfg = SACConfig(**cfg["metaworld"]["sac"])
    seq = make_task_sequence("metaworld", seed=cfg["seed"])
    if cfg.get("smoke"):
        seq = seq[:2]
        sac_cfg.batch_size = 32
        sac_cfg.learning_starts = 100
        sac_cfg.random_actions = 100
        cfg["budget_per_task"] = 1000

    obs_dim = cfg["metaworld"]["d_enc"]
    act_dim = cfg["metaworld"]["n_actions"]
    actor = build_actor(cfg["method"], "metaworld", cfg, total_tasks=len(seq))
    sac = SAC(actor, obs_dim=obs_dim, act_dim=act_dim, cfg=sac_cfg, device=device)
    rb = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, capacity=sac_cfg.buffer_size)

    per_task_returns: List[float] = []
    success_at_end: List[float] = []
    forgetting_vals: List[float] = []
    delta = int(cfg["budget_per_task"])

    for k, env_thunk in enumerate(seq):
        env = env_thunk()
        # Task boundary: freeze previous module, add new (CompoNet).
        if k > 0:
            if cfg["method"] == "componet":
                actor.add_new_task()
            elif cfg["method"] in ("prognet", "packnet"):
                actor["trunk"].add_new_task()
            sac.on_task_change(obs_dim=obs_dim, act_dim=act_dim)
            rb.reset()

        t0 = time.time()
        obs, _ = env.reset(seed=cfg["seed"] + k)
        ep_return = 0.0
        last_returns: List[float] = []
        for t in range(delta):
            if t < sac_cfg.random_actions:
                act = np.random.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
            else:
                act = sac.select_action(np.asarray(obs, dtype=np.float32))
            res = env.step(act)
            if len(res) == 5:
                next_obs, reward, terminated, truncated, info = res
                done = bool(terminated or truncated)
            else:  # legacy gym
                next_obs, reward, done, info = res
            rb.add(obs, act, reward, next_obs, done)
            ep_return += float(reward)
            obs = next_obs
            if done:
                last_returns.append(ep_return)
                obs, _ = env.reset()
                ep_return = 0.0
            # Updates.
            if t >= sac_cfg.learning_starts and rb.size >= sac_cfg.batch_size:
                batch = rb.sample(sac_cfg.batch_size)
                sac.update(batch)
        # End-of-task evaluation (use last 20 episodes' average return).
        avg_ret = float(np.mean(last_returns[-20:])) if last_returns else 0.0
        per_task_returns.append(avg_ret)
        # Use task-provided success metric when available (Meta-World v2 envs
        # expose `info["success"]` at episode end).
        succ = float(info.get("success", 0.0)) if isinstance(info, dict) else 0.0
        success_at_end.append(succ)
        print(
            f"[task {k}] avg_return={avg_ret:.3f}  success={succ:.3f}  "
            f"elapsed={time.time() - t0:.1f}s"
        )

    return {
        "sequence": "metaworld",
        "method": cfg["method"],
        "tasks": [t for t in METAWORLD_CW20_TASKS] * 2,
        "per_task_avg_return": per_task_returns,
        "success_at_end_of_task": success_at_end,
        "average_performance": average_performance(success_at_end),
    }


# ---------------------------------------------------------------------------
# ALE (PPO) trainer
# ---------------------------------------------------------------------------
def train_ale(cfg: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Run a SpaceInvaders / Freeway sequence with PPO."""
    ppo_cfg = PPOConfig(**cfg["ale"]["ppo"])
    if cfg.get("smoke"):
        ppo_cfg.num_envs = 2
        ppo_cfg.num_steps = 16
        ppo_cfg.num_minibatches = 2
        cfg["budget_per_task"] = 1024

    seq = make_task_sequence(cfg["sequence"], seed=cfg["seed"])
    if cfg.get("smoke"):
        seq = seq[:2]
    n_tasks = len(seq)
    success_scores = (
        SPACEINVADERS_SUCCESS_SCORES
        if cfg["sequence"] == "spaceinvaders"
        else FREEWAY_SUCCESS_SCORES
    )

    actor = build_actor(cfg["method"], cfg["sequence"], cfg, total_tasks=n_tasks)
    encoder = actor.encoder if hasattr(actor, "encoder") else actor["encoder"]
    n_actions = 6 if cfg["sequence"] == "spaceinvaders" else 3
    ppo = PPO(
        actor=actor,
        encoder=encoder,
        n_actions=n_actions,
        d_enc=cfg["ale"]["d_enc"],
        cfg=ppo_cfg,
        device=device,
    )

    per_task_returns: List[float] = []
    per_task_success: List[float] = []

    delta = int(cfg["budget_per_task"])
    rollout_T = ppo_cfg.num_steps

    for k, env_thunk in enumerate(seq):
        env = env_thunk()
        if k > 0:
            if cfg["method"] == "componet":
                actor.add_new_task()
            elif cfg["method"] in ("prognet", "packnet"):
                actor["trunk"].add_new_task()
            ppo.on_task_change()

        ppo._task_total_updates = max(1, delta // (rollout_T * ppo_cfg.num_envs))
        ppo._task_update_counter = 0

        obs, _ = env.reset(seed=cfg["seed"] + k)
        obs = np.asarray(obs, dtype=np.float32)
        steps_done = 0
        ep_return = 0.0
        recent_returns: List[float] = []

        # Storage for one rollout (single-env smoke loop -- the full impl
        # would use a SyncVectorEnv with 8 envs as in Table E.2).
        while steps_done < delta:
            obs_buf, act_buf, lp_buf, rew_buf, done_buf, val_buf = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for _ in range(rollout_T):
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                action, logp, value = ppo.select_action(obs_t)
                action_i = int(action.item())
                next_obs, reward, term, trunc, info = env.step(action_i)
                done = bool(term or trunc)
                obs_buf.append(obs)
                act_buf.append(action_i)
                lp_buf.append(float(logp.item()))
                rew_buf.append(float(reward))
                done_buf.append(float(done))
                val_buf.append(float(value.item()))
                ep_return += float(reward)
                obs = np.asarray(next_obs, dtype=np.float32)
                if done:
                    recent_returns.append(ep_return)
                    obs, _ = env.reset()
                    obs = np.asarray(obs, dtype=np.float32)
                    ep_return = 0.0
                steps_done += 1
                if steps_done >= delta:
                    break

            if not obs_buf:
                break
            obs_t = torch.as_tensor(
                np.stack(obs_buf), dtype=torch.float32, device=device
            )
            actions_t = torch.as_tensor(
                np.asarray(act_buf), dtype=torch.long, device=device
            )
            lps_t = torch.as_tensor(
                np.asarray(lp_buf), dtype=torch.float32, device=device
            )
            rews_t = torch.as_tensor(
                np.asarray(rew_buf), dtype=torch.float32, device=device
            ).unsqueeze(1)
            dones_t = torch.as_tensor(
                np.asarray(done_buf), dtype=torch.float32, device=device
            ).unsqueeze(1)
            vals_t = torch.as_tensor(
                np.asarray(val_buf), dtype=torch.float32, device=device
            ).unsqueeze(1)

            with torch.no_grad():
                last_obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                last_value = ppo.critic(ppo.encoder(last_obs_t)).squeeze()

            advs, rets = PPO.compute_gae(
                rews_t,
                vals_t,
                dones_t,
                last_value,
                gamma=ppo_cfg.gamma,
                lam=ppo_cfg.gae_lambda,
            )
            advs = advs.flatten()
            rets = rets.flatten()
            vals_flat = vals_t.flatten()
            ppo.update(obs_t, actions_t, lps_t, advs, rets, vals_flat)

        avg_ret = float(np.mean(recent_returns[-20:])) if recent_returns else 0.0
        per_task_returns.append(avg_ret)
        succ = 1.0 if avg_ret >= success_scores[k % len(success_scores)] else 0.0
        per_task_success.append(succ)
        print(f"[task {k}] avg_return={avg_ret:.3f}  success={succ:.3f}")

    return {
        "sequence": cfg["sequence"],
        "method": cfg["method"],
        "tasks_modes": (
            SPACEINVADERS_MODES if cfg["sequence"] == "spaceinvaders" else FREEWAY_MODES
        ),
        "per_task_avg_return": per_task_returns,
        "success_at_end_of_task": per_task_success,
        "average_performance": average_performance(per_task_success),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    cfg = merge_cli(load_config(args.config), args)
    cfg["smoke"] = args.smoke

    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    output_dir = Path(cfg.get("output_dir", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg["sequence"] == "metaworld":
        result = train_metaworld(cfg, device)
    elif cfg["sequence"] in ("spaceinvaders", "freeway"):
        result = train_ale(cfg, device)
    else:
        raise ValueError(f"Unknown sequence: {cfg['sequence']}")

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
