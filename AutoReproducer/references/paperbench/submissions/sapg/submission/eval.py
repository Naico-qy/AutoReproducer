#!/usr/bin/env python
"""SAPG evaluation: load checkpoint, roll out leader policy, report metrics.

Usage:
    python eval.py --config configs/default.yaml --ckpt runs/.../final.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from data.loader import make_env
from model.architecture import SAPGPolicySet
from utils.config import load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to .pt checkpoint. If omitted, evaluates random init.",
    )
    p.add_argument("--num_episodes", type=int, default=None)
    p.add_argument(
        "--policy_idx",
        type=int,
        default=0,
        help="Which policy to evaluate (0 = leader).",
    )
    p.add_argument("--deterministic", action="store_true")
    p.add_argument(
        "--output", type=str, default=None, help="Path to write eval metrics JSON."
    )
    return p.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.num_episodes is not None:
        cfg.eval.num_episodes = args.num_episodes
    if args.deterministic:
        cfg.eval.deterministic = True

    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")

    # We evaluate a SINGLE block-size of envs (one policy)
    block_size = cfg.env.num_envs // cfg.env.num_policies
    eval_envs = max(min(block_size, 1024), 8)
    env = make_env(
        task=cfg.experiment.task,
        num_envs=eval_envs,
        obs_dim=cfg.env.obs_dim,
        action_dim=cfg.env.action_dim,
        device=str(device),
        max_episode_length=cfg.env.max_episode_length,
        seed=cfg.experiment.seed + 1000,
    )

    policy = SAPGPolicySet(cfg.model, cfg.env).to(device)
    if args.ckpt and os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        policy.load_state_dict(ckpt["policy"], strict=False)
        print(f"[SAPG-eval] loaded {args.ckpt}")
    else:
        print(
            "[SAPG-eval] WARNING: no checkpoint loaded; reporting random-init metrics"
        )

    obs = env.reset()
    if not torch.is_tensor(obs):
        obs = torch.as_tensor(obs, device=device, dtype=torch.float32)

    n_episodes = int(cfg.eval.num_episodes)
    total_returns: list = []
    total_successes: list = []
    steps = 0
    max_steps = n_episodes * cfg.env.max_episode_length

    while steps < max_steps:
        action, _, _ = policy.actor.act(
            obs,
            policy_idx=int(args.policy_idx),
            deterministic=bool(cfg.eval.deterministic),
        )
        action = action.clamp(-1.0, 1.0)
        next_obs, rew, done, info = env.step(action)
        if not torch.is_tensor(next_obs):
            next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
        if isinstance(info, dict):
            if "episode_return" in info:
                total_returns.append(float(info["episode_return"].mean().item()))
            if "successes" in info:
                total_successes.append(float(info["successes"].mean().item()))
        obs = next_obs
        steps += 1

    metric_name = str(cfg.eval.metric)
    metrics = {
        "task": cfg.experiment.task,
        "policy_idx": int(args.policy_idx),
        "num_envs": eval_envs,
        "deterministic": bool(cfg.eval.deterministic),
        "metric": metric_name,
        "mean_episode_return": float(np.mean(total_returns)) if total_returns else 0.0,
        "mean_successes": float(np.mean(total_successes)) if total_successes else 0.0,
    }
    if metric_name == "successes":
        metrics["score"] = metrics["mean_successes"]
    else:
        metrics["score"] = metrics["mean_episode_return"]

    out_path = args.output or cfg.logging.metrics_path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[SAPG-eval] wrote {out_path}: {metrics}")
    except (PermissionError, OSError):
        fallback = "./eval_metrics.json"
        with open(fallback, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[SAPG-eval] wrote fallback {fallback}: {metrics}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
