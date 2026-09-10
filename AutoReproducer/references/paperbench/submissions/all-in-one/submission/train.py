"""Training entrypoint for the Simformer.

Usage:  python train.py --config configs/default.yaml [--task two_moons]

This implements the Simformer training loop (Gloeckler et al. ICML 2024,
§3.3), driven by per-batch sampling of the condition mask M_C from the
mixture distribution specified in addendum.md and standard denoising
score-matching against the Gaussian transition kernel of the chosen SDE.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from tqdm import tqdm

from data import build_loader
from model import Simformer, get_sde, simformer_loss
from tasks import get_task


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument(
        "--task", type=str, default=None, help="Override task name from config"
    )
    p.add_argument("--num_simulations", type=int, default=None)
    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def maybe_override(cfg: dict, args: argparse.Namespace) -> dict:
    if args.task:
        cfg["task"] = args.task
    if args.num_simulations:
        cfg["num_simulations"] = args.num_simulations
    if args.num_steps:
        cfg["num_steps"] = args.num_steps
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    return cfg


def cosine_lr(step: int, total: int, base: float, warmup: int = 1000) -> float:
    if step < warmup:
        return base * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def main() -> None:
    args = parse_args()
    cfg = maybe_override(load_config(args.config), args)
    torch.manual_seed(cfg.get("seed", 0))

    device = torch.device(args.device)
    output_dir = Path(cfg.get("output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Task & data
    task = get_task(cfg["task"])
    print(
        f"[Simformer] Task: {task.name}  d_theta={task.num_params}  d_x={task.num_data}"
    )

    loader, dataset = build_loader(
        task, cfg["num_simulations"], cfg["batch_size"], seed=cfg.get("seed", 0)
    )
    num_vars = task.num_params + task.num_data

    # Model
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
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Simformer] Model parameters: {n_params / 1e6:.2f}M")

    # SDE
    sde = get_sde(cfg["sde"])

    # Attention mask (from task graph if requested)
    attention_mask = None
    if cfg.get("use_structured_attention", False):
        adj = torch.tensor(task.structured_mask(), dtype=torch.bool, device=device)
        # PyTorch attention expects True = "block". Adjacency is "True = allowed".
        attention_mask = ~adj
        attention_mask.fill_diagonal_(False)  # always allow self
        print("[Simformer] Using structured attention mask from task graph.")

    # Optimizer
    base_lr = float(cfg["lr"])
    opt = AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
    )

    mask_probs = cfg["mask_distribution"]

    # Training loop
    total_steps = int(cfg["num_steps"])
    log_every = max(1, total_steps // 50)
    losses: list[float] = []
    step = 0
    t0 = time.time()
    pbar = tqdm(total=total_steps, desc="train")
    data_iter = iter(loader)
    while step < total_steps:
        try:
            x0 = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x0 = next(data_iter)
        x0 = x0.to(device)
        # LR schedule (cosine with warmup) -- standard for ICML diffusion work.
        lr = cosine_lr(step, total_steps, base_lr)
        for pg in opt.param_groups:
            pg["lr"] = lr

        loss = simformer_loss(
            model,
            x0,
            sde,
            num_params=task.num_params,
            mask_probs=mask_probs,
            attention_mask=attention_mask,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(float(loss.item()))
        step += 1
        pbar.update(1)
        if step % log_every == 0:
            pbar.set_postfix(loss=f"{losses[-1]:.4f}", lr=f"{lr:.2e}")

        if step % cfg.get("checkpoint_every", 5000) == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "norm_mean": dataset.mean,
                    "norm_std": dataset.std,
                },
                output_dir / f"ckpt_step{step}.pt",
            )
    pbar.close()

    # Final save
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": cfg,
            "norm_mean": dataset.mean,
            "norm_std": dataset.std,
        },
        output_dir / "ckpt_final.pt",
    )

    elapsed = time.time() - t0
    summary = {
        "task": task.name,
        "num_simulations": cfg["num_simulations"],
        "num_steps": total_steps,
        "final_loss": float(losses[-1]) if losses else None,
        "training_seconds": elapsed,
        "num_parameters": n_params,
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[Simformer] Training complete:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
