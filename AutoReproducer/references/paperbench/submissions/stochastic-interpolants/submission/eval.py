"""Evaluation entrypoint — Algorithm 2 of the paper.

Loads a trained checkpoint, samples from the probability flow ODE
(Algorithm 2 / Eq. 8) using either forward Euler or torchdiffeq's Dopri5,
and reports pixel-space metrics (PSNR / MSE).

Output schema follows the PaperBench reproduction-mode contract:
the metrics JSON is written to `--output` (default `/output/metrics.json`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional

import torch
import yaml

from interpolant import (
    StochasticInterpolant,
    sample_ode_dopri,
    sample_ode_euler,
)
from interpolant.coefficients import make_coefficients
from interpolant.couplings import make_coupling
from model import EMA, build_velocity_model
from data import build_dataloader
from utils import setup_logger
from utils.logging import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a stochastic-interpolant model.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, required=False, default=None)
    p.add_argument("--output", type=str, default="/output/metrics.json")
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument(
        "--solver", type=str, default=None, help="dopri5 | euler — overrides config."
    )
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve_cond_channels(task: str, image_channels: int = 3) -> int:
    return image_channels if task in ("inpainting", "superres") else 0


def _psnr_mse(x: torch.Tensor, y: torch.Tensor) -> tuple:
    """Both tensors in [-1, 1]; return (psnr_db, mse)."""
    mse = (x - y).pow(2).mean().item()
    if mse <= 0:
        return float("inf"), 0.0
    # PSNR for the [-1, 1] range has dynamic range = 2.
    psnr = 10.0 * math.log10((2.0**2) / mse)
    return psnr, mse


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.solver is not None:
        cfg["eval"]["solver"] = args.solver
    if args.num_samples is not None:
        cfg["eval"]["num_samples"] = int(args.num_samples)

    task = cfg["task"]
    out_path = Path(args.output)
    ensure_dir(str(out_path.parent))

    logger = setup_logger("eval")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Build model + load checkpoint
    # ------------------------------------------------------------------
    cond_channels = _resolve_cond_channels(task)
    model = build_velocity_model(
        cfg["model"], in_channels=3, cond_channels=cond_channels
    ).to(device)

    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        sd = torch.load(args.checkpoint, map_location=device)
        try:
            ema = EMA(model, decay=float(cfg["train"].get("ema_decay", 0.9999)))
            ema.load_state_dict(sd["ema"])
            ema.copy_to(model)
            logger.info(f"loaded EMA weights from {args.checkpoint}")
        except Exception:
            model.load_state_dict(sd["model"])
            logger.info(f"loaded raw model weights from {args.checkpoint}")
    else:
        logger.info("no checkpoint provided; evaluating randomly-initialised model")

    model.eval()

    # ------------------------------------------------------------------
    # Coupling + interpolant
    # ------------------------------------------------------------------
    coupling_cfg = dict(cfg["coupling"])
    coupling_cfg.setdefault("low_res_size", cfg["data"].get("low_res_size", 64))
    coupling = make_coupling(coupling_cfg)

    coeffs = make_coefficients(cfg["interpolant"]["schedule"])
    interp = StochasticInterpolant(coeffs)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if args.debug:
        cfg["data"]["debug_image_size"] = 32
    dl = build_dataloader(
        cfg["data"],
        split="validation",
        batch_size=int(cfg["eval"].get("batch_size", 8)),
        num_workers=int(cfg["data"].get("num_workers", 2)),
        debug=args.debug,
    )

    # ------------------------------------------------------------------
    # Sampling loop
    # ------------------------------------------------------------------
    solver = cfg["eval"].get("solver", "dopri5")
    rtol = float(cfg["eval"].get("rtol", 1e-5))
    atol = float(cfg["eval"].get("atol", 1e-5))
    n_steps_euler = int(cfg["eval"].get("num_steps_euler", 100))
    target = int(cfg["eval"].get("num_samples", 100))

    psnrs = []
    mses = []
    n_done = 0

    with torch.no_grad():
        for batch in dl:
            if n_done >= target:
                break
            x1 = batch["x1"].to(device)
            cls = batch["label"].to(device) if "label" in batch else None

            x0, xi = coupling.sample_x0(x1)
            mask = xi if task == "inpainting" else None

            def velocity_fn(x_t, t_b):
                return model(x_t, t_b, cond=xi, cls=cls, mask=mask)

            if solver == "euler":
                x_hat = sample_ode_euler(velocity_fn, x0, n_steps=n_steps_euler)
            else:
                x_hat = sample_ode_dopri(velocity_fn, x0, rtol=rtol, atol=atol)

            psnr, mse = _psnr_mse(x_hat.clamp(-1, 1), x1)
            psnrs.append(psnr)
            mses.append(mse)
            n_done += x1.shape[0]
            logger.info(
                f"[eval] batch {len(psnrs):>3} | psnr={psnr:.2f} dB | mse={mse:.4e}"
            )

    metrics = {
        "task": task,
        "solver": solver,
        "num_samples": int(n_done),
        "psnr_mean_db": float(sum(psnrs) / max(1, len(psnrs))),
        "mse_mean": float(sum(mses) / max(1, len(mses))),
        "psnr_per_batch_db": psnrs,
        "mse_per_batch": mses,
    }
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"wrote metrics to {out_path}")


if __name__ == "__main__":
    main()
