"""Evaluation entrypoint for the Simformer.

Loads a checkpoint, samples the posterior for a held-out observation,
computes C2ST against a reference posterior (closed-form for the
Gaussian-Linear task or task-specific simulator-based reference for the
others), and writes ``metrics.json`` to ``output_dir``.

Usage:
    python eval.py --config configs/default.yaml --checkpoint output/ckpt_final.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from data import simulate_dataset
from model import Simformer, get_sde, sample_conditional
from tasks import get_task
from utils import c2st_random_forest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, default="output/ckpt_final.pt")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--num_eval_obs", type=int, default=5)
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def reference_posterior_samples(
    task, x_obs_unnorm: torch.Tensor, num_samples: int
) -> torch.Tensor:
    """Generate reference posterior samples.

    For the Gaussian Linear task we have a closed-form posterior. For other
    tasks we fall back to importance sampling from the prior weighted by the
    likelihood (sample-then-resample); this is a faithful reference but slower.
    """
    if hasattr(task, "reference_posterior"):
        return task.reference_posterior(x_obs_unnorm, num_samples)
    # Importance sampling fallback
    n_pool = max(50_000, num_samples * 50)
    theta = task.prior(n_pool)
    x = task.simulator(theta)
    diff = (x - x_obs_unnorm.unsqueeze(0)).reshape(n_pool, -1)
    log_w = -(diff**2).sum(dim=-1)  # Gaussian-ish kernel
    w = torch.softmax(log_w, dim=0)
    idx = torch.multinomial(w, num_samples, replacement=True)
    return theta[idx]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir or cfg.get("output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    task = get_task(cfg["task"])
    num_vars = task.num_params + task.num_data

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mcfg = cfg["model"]
    model = Simformer(
        num_variables=num_vars,
        embedding_dim=mcfg["embedding_dim"],
        num_heads=mcfg["num_heads"],
        num_layers=mcfg["num_layers"],
        ff_dim=mcfg["ff_dim"],
        fourier_dim=mcfg["fourier_dim"],
        dropout=mcfg.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    norm_mean = ckpt["norm_mean"].to(device)
    norm_std = ckpt["norm_std"].to(device)

    sde = get_sde(cfg["sde"])

    # Generate eval observations from the simulator.
    torch.manual_seed(123)
    eval_theta = task.prior(args.num_eval_obs)
    eval_x = task.simulator(eval_theta)

    n_post = cfg["sampling"]["num_samples"]
    n_steps = cfg["sampling"]["num_steps"]

    c2st_scores: list[float] = []
    for i in range(args.num_eval_obs):
        x_obs = eval_x[i]
        # Posterior conditioning: parameters latent, data conditioned.
        cond_mask = torch.zeros(num_vars, dtype=torch.bool, device=device)
        cond_mask[task.num_params :] = True
        # Build observed values vector (in normalized space).
        joint_obs = torch.zeros(num_vars, device=device)
        joint_obs[task.num_params :] = x_obs.to(device)
        joint_obs_norm = (joint_obs - norm_mean.squeeze(0)) / norm_std.squeeze(0)

        samples_norm = sample_conditional(
            model,
            sde,
            condition_mask=cond_mask,
            observed_values=joint_obs_norm,
            num_steps=n_steps,
            num_samples=n_post,
        )
        # Denormalize
        samples = samples_norm * norm_std + norm_mean
        post_samples = samples[:, : task.num_params].cpu()

        # Reference posterior samples
        ref = reference_posterior_samples(task, x_obs.cpu(), n_post)

        c2st_val = c2st_random_forest(
            post_samples.numpy(),
            ref.numpy(),
            n_estimators=cfg["eval"]["c2st_n_estimators"],
        )
        c2st_scores.append(c2st_val)
        print(f"[eval] obs {i}: C2ST = {c2st_val:.3f}")

    metrics = {
        "task": task.name,
        "num_simulations": cfg["num_simulations"],
        "num_eval_obs": args.num_eval_obs,
        "c2st_scores": c2st_scores,
        "c2st_mean": float(np.mean(c2st_scores)),
        "c2st_std": float(np.std(c2st_scores)),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("[eval] Metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
