"""Evaluation entrypoint.

Loads a checkpoint produced by train.py and computes:

    - PINN loss L(w)         (Eq. 2)
    - L2 relative error      (Section 2.2 / between Eqs. 2 and 3)
    - Gradient norm ||∇L(w)||₂

Writes a JSON summary to <output>/metrics.json — this is the file
read by the PaperBench reproduction-grading harness.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import torch
import yaml

from data.loader import build_collocation_points, evaluation_grid
from model.architecture import PINN
from model.pdes import build_pde, pinn_loss
from utils.metrics import gradient_norm, l2_relative_error


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--checkpoint", required=True, help="Path to model.pt produced by train.py."
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Where to write metrics.json. Defaults to "
        "/output if it exists, else next to checkpoint.",
    )
    p.add_argument(
        "--pde", default=None, choices=[None, "convection", "reaction", "wave"]
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)

    # Load checkpoint and (potentially) override the config with the one
    # serialized inside the checkpoint.
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", cfg)

    pde_name = args.pde or cfg["pde"]["name"]
    pde_kwargs = cfg["pde"].get(pde_name, {})
    pde = build_pde(pde_name, **pde_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PINN(
        in_dim=cfg["model"]["in_dim"],
        out_dim=cfg["model"]["out_dim"],
        hidden_widths=cfg["model"]["hidden_widths"],
        activation=cfg["model"]["activation"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # PINN loss & gradient norm — at the same collocation points used
    # for training.
    points = build_collocation_points(
        pde,
        n_res=cfg["data"]["n_res"],
        n_ic=cfg["data"]["n_ic"],
        n_bc=cfg["data"]["n_bc"],
        grid_nx=cfg["data"]["grid_nx"],
        grid_nt=cfg["data"]["grid_nt"],
        seed=cfg["training"]["seed"],
    ).to(device)

    loss = pinn_loss(
        model,
        pde,
        x_res=points.x_res,
        x_ic=points.x_ic,
        x_bc_left=points.x_bc_left,
        x_bc_right=points.x_bc_right,
        x_ic_velocity=points.x_ic_velocity,
    )
    grad_norm = gradient_norm(loss, list(model.parameters()))

    # L2RE on the union of grid + IC + BC points (Section 2.2).
    grid = evaluation_grid(pde)
    coords = grid["all"].to(device)
    with torch.no_grad():
        pred = model(coords)
        target = pde.exact_fn(coords)
    l2re = l2_relative_error(pred, target)

    metrics: Dict[str, Any] = {
        "pde": pde_name,
        "loss": float(loss.detach()),
        "l2re": l2re,
        "gradient_norm": grad_norm,
        "num_params": model.num_params,
    }
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = (
            "/output" if os.path.isdir("/output") else os.path.dirname(args.checkpoint)
        )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
