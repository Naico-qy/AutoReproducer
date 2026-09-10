"""Training entrypoint — Algorithm 1 of the paper.

This implements the full stochastic-interpolant training loop with a
data-dependent coupling. Hyper-parameters come from Appendix B and the
addendum:

    * Adam (lr 2e-4, wd 0)
    * StepLR (γ=0.99 every 1000 steps)
    * Gradient norm clipping at 10,000
    * Batch size 32, 200 000 gradient steps
    * Lightning Fabric for parallelism / mixed precision

Algorithm 1:

    repeat
      for i in 1..n_b:
        x_1 ~ ρ_1, ζ ~ N(0, Id), t ~ U(0, 1)
        x_0 = m(x_1) + σ ζ
        I_t = α_t x_0 + β_t x_1   (γ_t = 0 for our experiments)
      L̂_b = n_b⁻¹ Σ |b̂(I_t)|² − 2 İ_t · b̂(I_t)
      backward + Adam step on L̂_b
    until converged
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import yaml

from interpolant import (
    StochasticInterpolant,
    velocity_loss,
)
from interpolant.coefficients import make_coefficients
from interpolant.couplings import make_coupling
from model import EMA, build_velocity_model
from data import build_dataloader
from utils import setup_logger, format_step
from utils.logging import ensure_dir


# ---------------------------------------------------------------------------
# CLI / config helpers ------------------------------------------------------
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a stochastic-interpolant velocity model."
    )
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument(
        "--task",
        type=str,
        default=None,
        help="Override config['task']: superres | inpainting",
    )
    p.add_argument(
        "--steps", type=int, default=None, help="Override the number of gradient steps."
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Use a tiny synthetic dataset for fast smoke-tests.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override config['train']['output_dir'].",
    )
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Coupling / model wiring ---------------------------------------------------
# ---------------------------------------------------------------------------


def _resolve_cond_channels(task: str, image_channels: int = 3) -> int:
    """How many extra channels does the network see?

    §4.1 in-painting: the mask ξ has 1 channel (broadcast across colour
        channels in our coupling, but stored as `image_channels` for
        convenience so the U-Net just sees a binary 3-channel mask).
    §4.2 super-resolution: ξ = U(D(x_1)) is a full RGB image —
        `image_channels` extra channels.
    Independent baseline: no conditioning ⇒ 0.
    """
    if task == "inpainting":
        return image_channels  # mask broadcast to all channels
    if task == "superres":
        return image_channels  # upsampled low-res image
    return 0


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.task is not None:
        cfg["task"] = args.task
    task = cfg["task"]
    cfg.setdefault("coupling", {}).setdefault("kind", task)

    if args.steps is not None:
        cfg["train"]["steps"] = int(args.steps)
    if args.output is not None:
        cfg["train"]["output_dir"] = args.output

    out_dir = Path(cfg["train"]["output_dir"])
    ensure_dir(str(out_dir))

    logger = setup_logger("train")
    logger.info(f"Running task={task}  out={out_dir}")

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if args.debug:
        cfg["data"]["debug_image_size"] = 32
    dl = build_dataloader(
        cfg["data"],
        split="train",
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["data"].get("num_workers", 4)),
        debug=args.debug,
    )

    # ------------------------------------------------------------------
    # Coupling, interpolant, velocity model
    # ------------------------------------------------------------------
    coupling_cfg = dict(cfg["coupling"])
    coupling_cfg.setdefault("low_res_size", cfg["data"].get("low_res_size", 64))
    coupling = make_coupling(coupling_cfg)

    coeffs = make_coefficients(cfg["interpolant"]["schedule"])
    interp = StochasticInterpolant(coeffs)

    cond_channels = _resolve_cond_channels(task)
    model = build_velocity_model(
        cfg["model"], in_channels=3, cond_channels=cond_channels
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"velocity model has {n_params / 1e6:.1f}M parameters")

    # ------------------------------------------------------------------
    # Optimiser + scheduler — Appendix B
    # ------------------------------------------------------------------
    optim = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    sched = torch.optim.lr_scheduler.StepLR(
        optim,
        step_size=int(cfg["train"]["step_size"]),
        gamma=float(cfg["train"]["gamma"]),
    )
    grad_clip = float(cfg["train"].get("grad_clip", 10000.0))

    ema = EMA(model, decay=float(cfg["train"].get("ema_decay", 0.9999)))

    # ------------------------------------------------------------------
    # Training loop — Algorithm 1
    # ------------------------------------------------------------------
    total_steps = int(cfg["train"]["steps"])
    log_every = int(cfg["train"].get("log_every", 100))
    ckpt_every = int(cfg["train"].get("ckpt_every", 5000))

    losses = []
    step = 0
    t0 = time.time()
    model.train()
    while step < total_steps:
        for batch in dl:
            if step >= total_steps:
                break
            x1 = batch["x1"].to(device, non_blocking=True)
            cls = (
                batch["label"].to(device, non_blocking=True)
                if "label" in batch
                else None
            )

            # ---- Algorithm 1: sample t, draw coupling, build I_t -----
            t = torch.rand(x1.shape[0], device=device, dtype=x1.dtype)
            x0, xi = coupling.sample_x0(x1)
            it, it_dot, _z = interp.build(x0, x1, t)

            # ---- forward / loss --------------------------------------
            mask = None
            if task == "inpainting" and xi is not None:
                # xi == keep mask in [0,1]^{C,H,W}
                mask = xi
            v_pred = model(
                it,
                t,
                cond=xi,
                cls=cls,
                mask=mask,
            )
            # Apply the §4.1 mask trick to the loss as well so we don't
            # waste capacity learning to predict zero on unmasked pixels.
            loss_mask = None
            if task == "inpainting" and mask is not None:
                loss_mask = 1.0 - mask

            loss = velocity_loss(v_pred, it_dot, mask=loss_mask)

            # ---- step ------------------------------------------------
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            sched.step()
            ema.update(model)

            losses.append(float(loss.detach().cpu()))
            step += 1

            if step % log_every == 0 or step == 1:
                lr_now = sched.get_last_lr()[0]
                avg = sum(losses[-log_every:]) / max(1, len(losses[-log_every:]))
                logger.info(format_step(step, total_steps, avg, lr_now))

            if step % ckpt_every == 0:
                _save_checkpoint(out_dir, model, ema, optim, sched, step, cfg)

    _save_checkpoint(out_dir, model, ema, optim, sched, step, cfg, name="last.pt")
    duration = time.time() - t0
    logger.info(f"Training done. {step} steps in {duration:.1f}s")

    summary = {
        "steps": step,
        "final_loss": losses[-1] if losses else None,
        "mean_loss_last_100": (
            sum(losses[-100:]) / max(1, len(losses[-100:])) if losses else None
        ),
        "task": task,
    }
    with open(out_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def _save_checkpoint(
    out_dir, model, ema, optim, sched, step, cfg, name: Optional[str] = None
):
    ensure_dir(str(out_dir))
    payload = {
        "step": step,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optim.state_dict(),
        "scheduler": sched.state_dict(),
        "config": cfg,
    }
    fname = name if name is not None else f"step_{step:07d}.pt"
    torch.save(payload, str(out_dir / fname))


if __name__ == "__main__":
    main()
