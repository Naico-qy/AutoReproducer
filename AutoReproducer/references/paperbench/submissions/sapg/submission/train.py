#!/usr/bin/env python
"""
SAPG training entrypoint.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/shadow_hand.yaml --smoke

References:
    Singla, Agarwal, Pathak. "SAPG: Split and Aggregate Policy Gradients."
    ICML 2024. https://sapg-rl.github.io
    PPO baseline: Schulman et al. arXiv:1707.06347 (verified via paper_search).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from data.loader import make_env
from model.architecture import SAPGPolicySet
from sapg.algorithm import SAPGAlgorithm
from utils.config import load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny env count + few iters, useful for CI / reproduce.sh",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--num_iterations",
        type=int,
        default=None,
        help="Override config.experiment.num_iterations",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to write metrics.json (overrides config)",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    if args.smoke or getattr(cfg.experiment, "smoke", False):
        cfg.experiment.smoke = True
        cfg.env.num_envs = max(cfg.env.num_policies * 16, 96)
        cfg.experiment.num_iterations = 3
        cfg.ppo.horizon_length = 4
        cfg.ppo.mini_epochs = 1
    if args.num_iterations is not None:
        cfg.experiment.num_iterations = args.num_iterations
    if args.device is not None:
        cfg.experiment.device = args.device
    if args.seed is not None:
        cfg.experiment.seed = args.seed
    if args.output is not None:
        cfg.logging.metrics_path = args.output

    device_str = cfg.experiment.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    set_seed(int(cfg.experiment.seed))
    print(
        f"[SAPG] device={device} task={cfg.experiment.task} "
        f"M={cfg.env.num_policies} N={cfg.env.num_envs} "
        f"smoke={cfg.experiment.smoke}"
    )

    # ------------------------------------------------------------------
    # Build env, policies, algorithm
    # ------------------------------------------------------------------
    env = make_env(
        task=cfg.experiment.task,
        num_envs=cfg.env.num_envs,
        obs_dim=cfg.env.obs_dim,
        action_dim=cfg.env.action_dim,
        device=str(device),
        max_episode_length=cfg.env.max_episode_length,
        seed=cfg.experiment.seed,
    )
    obs = env.reset()
    if not torch.is_tensor(obs):
        obs = torch.as_tensor(obs, device=device, dtype=torch.float32)

    policy = SAPGPolicySet(model_cfg=cfg.model, env_cfg=cfg.env)
    algo = SAPGAlgorithm(cfg=cfg, policy_set=policy, device=device)
    n_params = policy.num_parameters()
    print(f"[SAPG] policy params: {n_params:,}")

    # ------------------------------------------------------------------
    # Training loop (Algorithm 1)
    # ------------------------------------------------------------------
    log_dir = Path(cfg.experiment.log_dir) / cfg.experiment.name
    log_dir.mkdir(parents=True, exist_ok=True)
    history = []

    t0 = time.time()
    for it in range(int(cfg.experiment.num_iterations)):
        obs, roll_metrics = algo.collect(env, obs)
        upd_metrics = algo.update()
        algo.storage.reset()

        elapsed = time.time() - t0
        steps = (it + 1) * cfg.env.num_envs * cfg.ppo.horizon_length
        merged = {
            "iter": it,
            "steps": steps,
            "elapsed_sec": elapsed,
            **roll_metrics,
            **upd_metrics,
        }
        history.append(merged)

        if (
            it % int(cfg.logging.log_interval) == 0
            or it + 1 == cfg.experiment.num_iterations
        ):
            print(
                f"[iter {it:5d}] steps={steps:>10d} "
                f"ret={roll_metrics['ep_return']:7.3f} "
                f"succ={roll_metrics['successes']:6.2f} "
                f"actor={merged.get('actor_loss', 0.0):7.4f} "
                f"critic={merged.get('critic_loss', 0.0):7.4f} "
                f"kl={merged.get('approx_kl', 0.0):.4f} "
                f"lr={merged.get('lr', 0.0):.2e}"
            )

        if (it + 1) % int(cfg.logging.save_interval) == 0:
            ckpt = log_dir / f"ckpt_{it:06d}.pt"
            torch.save(
                {"policy": policy.state_dict(), "iter": it, "cfg_path": args.config},
                ckpt,
            )

    # ------------------------------------------------------------------
    # Final metrics dump (judge reads this)
    # ------------------------------------------------------------------
    final = history[-1] if history else {}
    metrics = {
        "task": cfg.experiment.task,
        "num_envs": cfg.env.num_envs,
        "num_policies": cfg.env.num_policies,
        "num_iterations": int(cfg.experiment.num_iterations),
        "final_ep_return": float(final.get("ep_return", 0.0)),
        "final_successes": float(final.get("successes", 0.0)),
        "final_actor_loss": float(final.get("actor_loss", 0.0)),
        "final_critic_loss": float(final.get("critic_loss", 0.0)),
        "final_approx_kl": float(final.get("approx_kl", 0.0)),
        "history": history,
    }

    out_path = cfg.logging.metrics_path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[SAPG] wrote metrics to {out_path}")
    except (PermissionError, OSError) as e:
        # Fall back to local path if /output not writable
        fallback = log_dir / "metrics.json"
        with open(fallback, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[SAPG] could not write to {out_path} ({e}); used {fallback}")

    final_ckpt = log_dir / "final.pt"
    torch.save(
        {
            "policy": policy.state_dict(),
            "cfg_path": args.config,
            "iter": int(cfg.experiment.num_iterations) - 1,
        },
        final_ckpt,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
