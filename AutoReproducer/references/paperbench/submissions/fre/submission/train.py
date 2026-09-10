"""FRE training entrypoint.

Implements Algorithm 1 of Frans et al. 2024:

    1. Train the FRE encoder/decoder until convergence
       (Equation 6 -- ELBO loss with beta=0.01 KL).
    2. Freeze the encoder, then train the IQL policy/critic/value
       conditioned on z (Section 4.3).

Usage:
    python train.py --config configs/default.yaml [--smoke]

`--smoke` shortens the run for quick verification on a CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from data import make_loader
from model import FREAgent
from model.reward_priors import sample_reward


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument(
        "--output", type=str, default=None, help="output dir (default: cfg.output_dir)"
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="run a tiny sanity check (used by reproduce.sh)",
    )
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
def evaluate_fre_loss(
    agent: FREAgent, loader, cfg: dict, device: torch.device, *, n_batches: int = 5
) -> float:
    """Quick held-out-style FRE loss for logging."""
    agent.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            s_e, s_d, r_e, r_d = build_fre_batch(loader, cfg, device)
            loss, _ = agent.fre.loss(s_e, r_e, s_d, r_d)
            losses.append(loss.item())
    agent.train()
    return float(np.mean(losses))


def build_fre_batch(loader, cfg: dict, device: torch.device):
    """Sample a batch of reward functions and (s_e, s_d, r_e, r_d) tuples."""
    B = cfg["batch_size"]
    K = cfg["K_encode"]
    Kp = cfg["K_decode"]
    sd = loader.state_dim

    s_e = loader.sample_states(B * K).view(B, K, sd)
    s_d = loader.sample_states(B * Kp).view(B, Kp, sd)
    r_e = torch.zeros(B, K, device=device)
    r_d = torch.zeros(B, Kp, device=device)

    # AntMaze: linear rewards skip XY dims; goal rewards use only XY.
    skip = slice(0, 2) if cfg.get("domain") == "antmaze" else None
    goal_mask = slice(0, 2) if cfg.get("domain") == "antmaze" else None
    threshold = (
        cfg["goal_threshold_antmaze"]
        if cfg.get("domain") == "antmaze"
        else cfg["goal_threshold_exorl"]
    )

    goal_pool = loader.goal_pool

    for i in range(B):
        eta = sample_reward(
            cfg["prior"],
            sd,
            goal_pool=goal_pool,
            goal_threshold=threshold,
            goal_mask=goal_mask,
            linear_skip=skip,
            device=device,
        )
        with torch.no_grad():
            r_e[i] = eta(s_e[i])
            r_d[i] = eta(s_d[i])
    return s_e.to(device), s_d.to(device), r_e, r_d


# ---------------------------------------------------------------------------
def train_encoder(
    agent: FREAgent, loader, cfg: dict, device: torch.device, output_dir: Path
):
    """Phase 1 of Algorithm 1: train encoder + decoder jointly."""
    opt = torch.optim.Adam(
        list(agent.fre.encoder.parameters()) + list(agent.fre.decoder.parameters()),
        lr=cfg["lr"],
    )
    n_steps = cfg["encoder_steps"]
    log_every = cfg["log_every"]
    metrics = []
    t0 = time.time()
    for step in range(1, n_steps + 1):
        s_e, s_d, r_e, r_d = build_fre_batch(loader, cfg, device)
        loss, info = agent.fre.loss(s_e, r_e, s_d, r_d)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == 1:
            print(
                f"[encoder] step {step}/{n_steps}  "
                f"loss={info['loss']:.4f}  recon={info['recon_mse']:.4f}  "
                f"kl={info['kl']:.4f}  ({time.time() - t0:.1f}s)"
            )
            metrics.append({"phase": "encoder", "step": step, **info})
    with open(output_dir / "encoder_log.json", "w") as f:
        json.dump(metrics, f, indent=2)


# ---------------------------------------------------------------------------
def train_policy(
    agent: FREAgent, loader, cfg: dict, device: torch.device, output_dir: Path
):
    """Phase 2 of Algorithm 1: freeze encoder, train IQL on FRE-encoded z."""
    for p in agent.fre.encoder.parameters():
        p.requires_grad_(False)
    for p in agent.fre.decoder.parameters():
        p.requires_grad_(False)
    agent.fre.encoder.eval()
    agent.fre.decoder.eval()

    actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=cfg["lr"])
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=cfg["lr"])
    value_opt = torch.optim.Adam(agent.value.parameters(), lr=cfg["lr"])

    n_steps = cfg["policy_steps"]
    log_every = cfg["log_every"]
    skip = slice(0, 2) if cfg.get("domain") == "antmaze" else None
    goal_mask = slice(0, 2) if cfg.get("domain") == "antmaze" else None
    threshold = (
        cfg["goal_threshold_antmaze"]
        if cfg.get("domain") == "antmaze"
        else cfg["goal_threshold_exorl"]
    )
    metrics = []
    t0 = time.time()

    for step in range(1, n_steps + 1):
        batch = loader.sample_batch(cfg["batch_size"])
        # sample one fresh reward function per sample in the batch
        K = cfg["K_encode"]
        sd = loader.state_dim
        s_e = (
            loader.sample_states(cfg["batch_size"] * K)
            .view(cfg["batch_size"], K, sd)
            .to(device)
        )
        r_e = torch.zeros(cfg["batch_size"], K, device=device)
        r_train = torch.zeros(cfg["batch_size"], device=device)

        for i in range(cfg["batch_size"]):
            eta = sample_reward(
                cfg["prior"],
                sd,
                goal_pool=loader.goal_pool,
                goal_threshold=threshold,
                goal_mask=goal_mask,
                linear_skip=skip,
                device=device,
            )
            with torch.no_grad():
                r_e[i] = eta(s_e[i])
                r_train[i] = eta(batch.s[i : i + 1])[0]

        with torch.no_grad():
            z = agent.fre.encode(s_e, r_e)

        b = {
            "s": batch.s,
            "a": batch.a,
            "s_next": batch.s_next,
            "r": r_train,
            "done": batch.done,
        }
        v_loss, q_loss, a_loss, info = agent.compute_iql_loss(b, z)

        value_opt.zero_grad(set_to_none=True)
        v_loss.backward()
        value_opt.step()
        critic_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        critic_opt.step()
        actor_opt.zero_grad(set_to_none=True)
        a_loss.backward()
        actor_opt.step()
        agent.soft_update()

        if step % log_every == 0 or step == 1:
            print(
                f"[policy] step {step}/{n_steps}  "
                f"v={info['v_loss']:.3f}  q={info['q_loss']:.3f}  "
                f"a={info['a_loss']:.3f}  ({time.time() - t0:.1f}s)"
            )
            metrics.append({"phase": "policy", "step": step, **info})

    with open(output_dir / "policy_log.json", "w") as f:
        json.dump(metrics, f, indent=2)


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    output_dir = Path(args.output or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        cfg["batch_size"] = 16
        cfg["encoder_steps"] = 200
        cfg["policy_steps"] = 200
        cfg["log_every"] = 50

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    loader = make_loader(cfg["domain"], dataset=cfg.get("dataset"), device=str(device))
    print(
        f"Loaded {type(loader).__name__}  state_dim={loader.state_dim}  "
        f"action_dim={loader.action_dim}  N={len(loader)}"
    )

    agent = FREAgent(
        state_dim=loader.state_dim,
        action_dim=loader.action_dim,
        z_dim=cfg["z_dim"],
        n_reward_bins=cfg["n_reward_bins"],
        state_embed_dim=cfg["state_embed_dim"],
        reward_embed_dim=cfg["reward_embed_dim"],
        token_dim=cfg["token_dim"],
        enc_mlp_dims=cfg["encoder_layers"],
        dec_hidden=cfg["decoder_layers"],
        rl_hidden=cfg["rl_layers"],
        num_heads=cfg["encoder_attn_heads"],
        beta_kl=cfg["beta_kl"],
        discount=cfg["discount"],
        expectile=cfg["iql_expectile"],
        awr_temperature=cfg["awr_temperature"],
        target_tau=cfg["target_update_rate"],
    ).to(device)

    print("Phase 1: encoder pretraining")
    train_encoder(agent, loader, cfg, device, output_dir)

    print("Phase 2: IQL policy training (encoder frozen)")
    train_policy(agent, loader, cfg, device, output_dir)

    ckpt = output_dir / "fre_agent.pt"
    torch.save({"state_dict": agent.state_dict(), "cfg": cfg}, ckpt)
    print(f"Saved checkpoint to {ckpt}")


if __name__ == "__main__":
    main()
