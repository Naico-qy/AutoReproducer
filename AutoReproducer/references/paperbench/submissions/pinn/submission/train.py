"""Training entrypoint for "Challenges in Training PINNs".

This script implements the full Adam → L-BFGS → NNCG pipeline of
Rathore et al. (2024).  It supports the three test PDEs (convection,
reaction, wave) defined in Appendix A of the paper.

Usage
-----
    python train.py --config configs/default.yaml \
                    --pde convection \
                    --output_dir runs/convection

The `--pde` flag overrides `pde.name` in the YAML config and selects
the recommended Adam learning rate / seed reported in the addendum:

    convection : adam_lr=1e-4, seed=345
    reaction   : adam_lr=1e-3, seed=456
    wave       : adam_lr=1e-3, seed=567
"""

from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy
from typing import Any, Dict

import torch
import yaml

from data.loader import build_collocation_points, evaluation_grid
from model.architecture import PINN
from model.pdes import build_pde, pinn_loss
from optim.nncg import NysNewtonCG
from utils.metrics import gradient_norm, l2_relative_error
from utils.seed import set_seed


# Best per-PDE Adam LR / seed found by the authors (addendum).
PDE_RECOMMENDED = {
    "convection": dict(adam_lr=1.0e-4, seed=345),
    "reaction": dict(adam_lr=1.0e-3, seed=456),
    "wave": dict(adam_lr=1.0e-3, seed=567),
}


# ----------------------------------------------------------------------
# Config helpers
# ----------------------------------------------------------------------


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--pde", default=None, choices=[None, "convection", "reaction", "wave"]
    )
    p.add_argument("--output_dir", default="runs/pinn")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--adam_lr", type=float, default=None)
    p.add_argument("--total_iters", type=int, default=None)
    p.add_argument("--adam_iters", type=int, default=None)
    p.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override hidden_widths to (width, width, width).",
    )
    p.add_argument("--no_nncg", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny-iteration smoke run for CI / paperbench.",
    )
    return p.parse_args()


# ----------------------------------------------------------------------
# Phase 1: Adam
# ----------------------------------------------------------------------


def run_adam(model, pde, points, n_iters, lr, betas, eps, log_every, history, device):
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=tuple(betas), eps=eps)
    pts = points.to(device)
    for k in range(n_iters):
        opt.zero_grad(set_to_none=True)
        loss = pinn_loss(
            model,
            pde,
            x_res=pts.x_res,
            x_ic=pts.x_ic,
            x_bc_left=pts.x_bc_left,
            x_bc_right=pts.x_bc_right,
            x_ic_velocity=pts.x_ic_velocity,
        )
        loss.backward()
        opt.step()
        if k % log_every == 0 or k == n_iters - 1:
            history.append({"phase": "adam", "iter": k, "loss": float(loss.detach())})
            print(f"[adam {k:6d}] loss={float(loss):.3e}")
    return float(loss.detach())


# ----------------------------------------------------------------------
# Phase 2: L-BFGS with strong Wolfe line search
# ----------------------------------------------------------------------


def run_lbfgs(model, pde, points, n_iters, cfg, log_every, history, device):
    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=cfg["lbfgs_lr"],
        max_iter=cfg["lbfgs_max_iter"],
        history_size=cfg["lbfgs_history_size"],
        tolerance_grad=cfg["lbfgs_tolerance_grad"],
        tolerance_change=cfg["lbfgs_tolerance_change"],
        line_search_fn=cfg["lbfgs_line_search"],
    )
    pts = points.to(device)

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = pinn_loss(
            model,
            pde,
            x_res=pts.x_res,
            x_ic=pts.x_ic,
            x_bc_left=pts.x_bc_left,
            x_bc_right=pts.x_bc_right,
            x_ic_velocity=pts.x_ic_velocity,
        )
        loss.backward()
        return loss

    last_loss = None
    for k in range(n_iters):
        loss = opt.step(closure)
        last_loss = float(loss)
        if k % log_every == 0 or k == n_iters - 1:
            history.append({"phase": "lbfgs", "iter": k, "loss": last_loss})
            print(f"[lbfgs {k:6d}] loss={last_loss:.3e}")
    return last_loss


# ----------------------------------------------------------------------
# Phase 3: NNCG
# ----------------------------------------------------------------------


def run_nncg(model, pde, points, n_iters, cfg, log_every, history, device):
    opt = NysNewtonCG(
        model.parameters(),
        lr=cfg["lr"],
        mu=cfg["mu"],
        sketch_size=cfg["sketch_size"],
        update_freq=cfg["update_freq"],
        cg_tol=cfg["cg_tol"],
        cg_max_iter=cfg["cg_max_iter"],
        armijo_alpha=cfg["armijo_alpha"],
        armijo_beta=cfg["armijo_beta"],
    )
    pts = points.to(device)

    def closure():
        opt.zero_grad(set_to_none=True)
        return pinn_loss(
            model,
            pde,
            x_res=pts.x_res,
            x_ic=pts.x_ic,
            x_bc_left=pts.x_bc_left,
            x_bc_right=pts.x_bc_right,
            x_ic_velocity=pts.x_ic_velocity,
        )

    last_loss = None
    for k in range(n_iters):
        loss = opt.step(closure)
        last_loss = float(loss)
        if k % log_every == 0 or k == n_iters - 1:
            history.append({"phase": "nncg", "iter": k, "loss": last_loss})
            print(f"[nncg  {k:6d}] loss={last_loss:.3e}")
    return last_loss


# ----------------------------------------------------------------------
# Evaluation helper (matches eval.py output schema)
# ----------------------------------------------------------------------


def evaluate(model, pde, device) -> Dict[str, float]:
    grid = evaluation_grid(pde)
    coords = grid["all"].to(device)
    with torch.no_grad():
        pred = model(coords)
        target = pde.exact_fn(coords)
    return {
        "l2re": l2_relative_error(pred, target),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    pde_name = args.pde or cfg["pde"]["name"]
    cfg["pde"]["name"] = pde_name

    rec = PDE_RECOMMENDED[pde_name]
    if args.adam_lr is None:
        cfg["training"]["adam_lr"] = rec["adam_lr"]
    else:
        cfg["training"]["adam_lr"] = args.adam_lr
    if args.seed is None:
        cfg["training"]["seed"] = rec["seed"]
    else:
        cfg["training"]["seed"] = args.seed
    if args.total_iters is not None:
        cfg["training"]["total_iters"] = args.total_iters
    if args.adam_iters is not None:
        cfg["training"]["adam_iters"] = args.adam_iters
    if args.width is not None:
        cfg["model"]["hidden_widths"] = [args.width, args.width, args.width]
    if args.smoke:
        cfg["training"]["adam_iters"] = 50
        cfg["training"]["total_iters"] = 100
        cfg["training"]["lbfgs_iters"] = 50
        cfg["nncg"]["iters"] = 10
        cfg["nncg"]["cg_max_iter"] = 20

    cfg["training"]["lbfgs_iters"] = (
        cfg["training"]["total_iters"] - cfg["training"]["adam_iters"]
    )

    set_seed(cfg["training"]["seed"])
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device(
        cfg["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"[train] pde={pde_name} device={device} seed={cfg['training']['seed']}")
    print(
        f"[train] adam_lr={cfg['training']['adam_lr']}  "
        f"width={cfg['model']['hidden_widths'][0]}  "
        f"iters total={cfg['training']['total_iters']} "
        f"adam={cfg['training']['adam_iters']} "
        f"lbfgs={cfg['training']['lbfgs_iters']}"
    )

    # PDE spec
    pde_kwargs = cfg["pde"].get(pde_name, {})
    pde = build_pde(pde_name, **pde_kwargs)

    # Collocation points
    points = build_collocation_points(
        pde,
        n_res=cfg["data"]["n_res"],
        n_ic=cfg["data"]["n_ic"],
        n_bc=cfg["data"]["n_bc"],
        grid_nx=cfg["data"]["grid_nx"],
        grid_nt=cfg["data"]["grid_nt"],
        seed=cfg["training"]["seed"],
    )

    # Model
    model = PINN(
        in_dim=cfg["model"]["in_dim"],
        out_dim=cfg["model"]["out_dim"],
        hidden_widths=cfg["model"]["hidden_widths"],
        activation=cfg["model"]["activation"],
    ).to(device)
    print(f"[train] model has {model.num_params:,} parameters")

    # ----------------------------------------------------------------
    history = []
    metrics: Dict[str, Any] = {"pde": pde_name, "config": deepcopy(cfg)}

    t0 = time.time()
    # Phase 1: Adam
    metrics["loss_after_adam"] = run_adam(
        model,
        pde,
        points,
        n_iters=cfg["training"]["adam_iters"],
        lr=cfg["training"]["adam_lr"],
        betas=cfg["training"]["adam_betas"],
        eps=cfg["training"]["adam_eps"],
        log_every=cfg["training"]["log_every"],
        history=history,
        device=device,
    )
    metrics["l2re_after_adam"] = evaluate(model, pde, device)["l2re"]
    metrics["time_adam_sec"] = time.time() - t0

    # Phase 2: L-BFGS
    t1 = time.time()
    metrics["loss_after_lbfgs"] = run_lbfgs(
        model,
        pde,
        points,
        n_iters=cfg["training"]["lbfgs_iters"],
        cfg=cfg["training"],
        log_every=cfg["training"]["log_every"],
        history=history,
        device=device,
    )
    metrics["l2re_after_lbfgs"] = evaluate(model, pde, device)["l2re"]
    metrics["time_lbfgs_sec"] = time.time() - t1

    # Phase 3: NNCG (optional)
    if cfg["nncg"]["enabled"] and not args.no_nncg:
        t2 = time.time()
        metrics["loss_after_nncg"] = run_nncg(
            model,
            pde,
            points,
            n_iters=cfg["nncg"]["iters"],
            cfg=cfg["nncg"],
            log_every=cfg["training"]["log_every"],
            history=history,
            device=device,
        )
        metrics["l2re_after_nncg"] = evaluate(model, pde, device)["l2re"]
        metrics["time_nncg_sec"] = time.time() - t2

    # Final gradient norm at returned solution.
    pts_dev = points.to(device)
    final_loss = pinn_loss(
        model,
        pde,
        x_res=pts_dev.x_res,
        x_ic=pts_dev.x_ic,
        x_bc_left=pts_dev.x_bc_left,
        x_bc_right=pts_dev.x_bc_right,
        x_ic_velocity=pts_dev.x_ic_velocity,
    )
    metrics["final_loss"] = float(final_loss.detach())
    metrics["final_grad_norm"] = gradient_norm(final_loss, list(model.parameters()))
    metrics["final_l2re"] = evaluate(model, pde, device)["l2re"]
    metrics["history"] = history
    metrics["total_time_sec"] = time.time() - t0

    # Save artifacts
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": cfg},
        os.path.join(args.output_dir, "model.pt"),
    )
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    print(
        f"[train] done. final_loss={metrics['final_loss']:.3e} "
        f"final_l2re={metrics['final_l2re']:.3e}"
    )
    print(f"[train] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
