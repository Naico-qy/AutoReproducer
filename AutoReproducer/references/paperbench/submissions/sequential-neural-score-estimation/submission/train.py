"""Training entry-point for NPSE / TSNPSE (Sharrock et al., ICML 2024).

Implements the full TSNPSE algorithm (paper Algorithm 1):

    For r = 1, ..., R:
        For i = 1, ..., M:
            θ_i ~ p̄^{r-1}(θ),   x_i ~ p(x | θ_i)        (round-r simulations)
            (D ← D ∪ {(θ_i, x_i)})
        Train s_ψ(θ_t, x, t) by minimising  J_post^{TSNPSE-DSM}  on D.
        Compute p̄^r(θ) ∝ p(θ) · 1{ θ ∈ HPR_ε( p_ψ^r(θ|x_obs) ) }
                                                   (paper Appendix E.3.3)

Hyperparameters all match the paper (Appendix E.3 / Section 5.1):
    * 3-layer 256-hidden SiLU MLPs for θ, x embeddings (max(30, 4d) / 4p output)
    * 64-dim sinusoidal time embedding (Vaswani-style, paper Eq. (108))
    * Adam optimiser, lr = 1e-4
    * 15% validation hold-out, early stop after 1000 steps without improvement
    * Max 3000 training iters per round
    * Batch size 50 (M ∈ {1000, 10000}, non-sequential), 200 (sequential), 500 (M=100000)
    * R = 10 sequential rounds
    * VE SDE: σ_min ∈ {0.01, 0.05}, σ_max via Technique 1 (first-round data only)
    * VP SDE: β_min = 0.1, β_max = 11.0
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch import nn

from data import get_task
from model import (
    ScoreNetwork,
    VESDE,
    VPSDE,
    denoising_score_matching_loss,
)
from model.sampler import sample_probability_flow
from model.truncation import TruncatedProposal, make_round_proposal
from utils import compute_sigma_max_technique1


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    task: str = "two_moons"
    method: str = "tsnpse"  # one of {"npse", "tsnpse"}
    sde: str = "ve"  # one of {"ve", "vp"}
    simulation_budget: int = 1000
    n_rounds: int = 10
    max_iters_per_round: int = 3000
    early_stop_patience: int = 1000
    val_fraction: float = 0.15
    batch_size: Optional[int] = None  # auto from budget if None
    learning_rate: float = 1e-4
    sigma_min: float = 0.01  # 0.01 for 2-D, 0.05 elsewhere
    sigma_max: Optional[float] = None  # auto via Technique 1 if None
    beta_min: float = 0.1
    beta_max: float = 11.0
    epsilon_hpr: float = 5e-4
    n_posterior_samples: int = 20000
    observation_idx: int = 1
    seed: int = 1
    output_dir: str = "outputs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_checkpoint: bool = True
    log_every: int = 100


def _autoselect_batch_size(budget: int, sequential: bool) -> int:
    if budget >= 100000:
        return 500
    if sequential:
        return 200
    return 50


def load_config(path: Optional[str]) -> TrainConfig:
    cfg = TrainConfig()
    if path is None:
        return cfg
    with open(path, "r") as f:
        d = yaml.safe_load(f) or {}
    for k, v in d.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Single-round trainer (paper Section 5.1, Appendix E.3.2)
# ---------------------------------------------------------------------------
def train_score_network(
    score_net: ScoreNetwork,
    sde,
    theta: torch.Tensor,
    x: torch.Tensor,
    cfg: TrainConfig,
    importance_weights: Optional[torch.Tensor] = None,
) -> dict:
    """Train `score_net` on (theta, x) with the DSM loss until early-stop.

    Returns a dict with the best validation loss seen.
    """
    device = torch.device(cfg.device)
    score_net.to(device).train()

    n = theta.shape[0]
    n_val = max(int(cfg.val_fraction * n), 1)
    perm = torch.randperm(n)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    theta_tr, x_tr = theta[tr_idx].to(device), x[tr_idx].to(device)
    theta_val, x_val = theta[val_idx].to(device), x[val_idx].to(device)

    if importance_weights is not None:
        iw_tr = importance_weights[tr_idx].to(device)
        iw_val = importance_weights[val_idx].to(device)
    else:
        iw_tr = iw_val = None

    bsz = cfg.batch_size or _autoselect_batch_size(
        cfg.simulation_budget,
        sequential=(cfg.method.startswith("t") or cfg.method.startswith("s")),
    )
    bsz = min(bsz, theta_tr.shape[0])

    optim = torch.optim.Adam(score_net.parameters(), lr=cfg.learning_rate)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in score_net.state_dict().items()}
    no_improve = 0

    for it in range(cfg.max_iters_per_round):
        idx = torch.randint(0, theta_tr.shape[0], (bsz,), device=device)
        theta_b, x_b = theta_tr[idx], x_tr[idx]
        iw_b = iw_tr[idx] if iw_tr is not None else None

        optim.zero_grad()
        loss = denoising_score_matching_loss(
            score_net, sde, theta_b, x_b, importance_weights=iw_b
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(score_net.parameters(), 5.0)
        optim.step()

        # Validation loss after each step (paper Appendix E.3.2: "we compute
        # the loss function on these samples after each training step").
        score_net.eval()
        with torch.no_grad():
            val_loss = denoising_score_matching_loss(
                score_net, sde, theta_val, x_val, importance_weights=iw_val
            ).item()
        score_net.train()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {
                k: v.detach().clone() for k, v in score_net.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1

        if it % cfg.log_every == 0:
            print(
                f"      iter {it:5d}  train {loss.item():.4f}  "
                f"val {val_loss:.4f}  best {best_val:.4f}"
            )

        if no_improve >= cfg.early_stop_patience:
            print(f"    early-stopping at iter {it}; best val = {best_val:.4f}")
            break

    score_net.load_state_dict(best_state)
    return {"best_val_loss": best_val}


# ---------------------------------------------------------------------------
# Full training loop (paper Algorithm 1)
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train (T)SNPSE on an SBI benchmark task."
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config file.")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--method", type=str, default=None, choices=["npse", "tsnpse"])
    parser.add_argument("--sde", type=str, default=None, choices=["ve", "vp"])
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="Override cfg.max_iters_per_round (smoke runs).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.task:
        cfg.task = args.task
    if args.method:
        cfg.method = args.method
    if args.sde:
        cfg.sde = args.sde
    if args.budget:
        cfg.simulation_budget = args.budget
    if args.rounds:
        cfg.n_rounds = args.rounds
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.max_iters:
        cfg.max_iters_per_round = args.max_iters

    torch.manual_seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Task / simulator ------------------------------------------------
    task = get_task(cfg.task)
    print(f"[train] task = {task.name}  d = {task.theta_dim}  p = {task.x_dim}")
    x_obs = task.observation(cfg.observation_idx)
    device = torch.device(cfg.device)

    sequential = cfg.method == "tsnpse"
    R = cfg.n_rounds if sequential else 1
    M = cfg.simulation_budget // R

    # σ_min: 0.01 for 2-D experiments (SIR, Two Moons), 0.05 otherwise.
    if task.theta_dim <= 2 and cfg.task in ("two_moons", "sir"):
        cfg.sigma_min = 0.01
    elif cfg.sigma_min is None or cfg.sigma_min == 0.01:
        cfg.sigma_min = 0.05 if task.theta_dim > 2 else cfg.sigma_min

    # ---- Score network ---------------------------------------------------
    score_net = ScoreNetwork(theta_dim=task.theta_dim, x_dim=task.x_dim).to(device)
    print(f"[train] score net params: {score_net.num_parameters():,}")

    # ---- SDE -------------------------------------------------------------
    sde = None  # set after first round when sigma_max is known.

    truncated_props: list[TruncatedProposal] = []
    dataset_theta: list[torch.Tensor] = []
    dataset_x: list[torch.Tensor] = []

    for r in range(1, R + 1):
        print(f"\n=== Round {r}/{R} (M = {M} simulations) ===")

        # 1) draw simulations from the round-r proposal -----------------------
        if sequential and r > 1:
            sampler = make_round_proposal(truncated_props, task.prior_sample)
            theta_r = sampler(M).to(device)
        else:
            theta_r = task.prior_sample(M).to(device)
        x_r = task.simulate(theta_r.cpu()).to(device)
        dataset_theta.append(theta_r.cpu())
        dataset_x.append(x_r.cpu())

        all_theta = torch.cat(dataset_theta, dim=0)
        all_x = torch.cat(dataset_x, dim=0)

        # 2) standardisation buffers (paper Appendix E.3.3) ------------------
        score_net.set_normalization(
            all_theta.mean(0),
            all_theta.std(0).clamp_min(1e-6),
            all_x.mean(0),
            all_x.std(0).clamp_min(1e-6),
        )

        # 3) instantiate / update SDE (sigma_max from round-1 data only) -----
        if sde is None:
            if cfg.sde == "ve":
                if cfg.sigma_max is None:
                    sigma_max = compute_sigma_max_technique1(all_theta)
                    sigma_max = max(sigma_max, cfg.sigma_min * 10.0)
                else:
                    sigma_max = cfg.sigma_max
                print(
                    f"[train] σ_max = {sigma_max:.4f} (Technique 1 of Song & Ermon 2020)"
                )
                sde = VESDE(sigma_min=cfg.sigma_min, sigma_max=sigma_max).to(device)
            else:
                sde = VPSDE(beta_min=cfg.beta_min, beta_max=cfg.beta_max).to(device)

        # 4) train score network on D -----------------------------------------
        info = train_score_network(score_net, sde, all_theta, all_x, cfg)
        print(f"  round {r} best val loss = {info['best_val_loss']:.4f}")

        # 5) compute the truncated proposal p̄^r for the next round ----------
        if sequential and r < R:
            tprop = TruncatedProposal(
                score_net=score_net,
                sde=sde,
                x_obs=x_obs,
                prior_sampler=task.prior_sample,
                prior_log_prob=getattr(task, "prior_log_prob", None),
                n_posterior_samples=min(cfg.n_posterior_samples, 5000),
                epsilon=cfg.epsilon_hpr,
                device=device,
            )
            try:
                tprop.fit()
                truncated_props.append(tprop)
            except Exception as exc:  # noqa: BLE001
                # If HPR fitting fails (e.g. unstable network early on), skip
                # truncation for this round and fall back to the prior.
                print(
                    f"  [warn] HPR fit failed at round {r}: {exc}; falling back to prior."
                )

    # ---- Save final checkpoint ------------------------------------------
    if cfg.save_checkpoint:
        ckpt_path = out / "score_net.pt"
        torch.save(
            {
                "config": cfg.__dict__,
                "state_dict": score_net.state_dict(),
                "sde": cfg.sde,
                "sigma_min": cfg.sigma_min,
                "sigma_max": getattr(sde, "sigma_max", None),
                "beta_min": cfg.beta_min,
                "beta_max": cfg.beta_max,
                "task": task.name,
                "theta_dim": task.theta_dim,
                "x_dim": task.x_dim,
            },
            ckpt_path,
        )
        print(f"[train] saved checkpoint to {ckpt_path}")

    # Persist a tiny config summary.
    with open(out / "train_config.json", "w") as f:
        json.dump(
            {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
