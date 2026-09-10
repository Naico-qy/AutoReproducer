"""
Train the convolutional VAE used in Section 5.3.

The schedule follows the addendum exactly:
    Optimizer       : Adam
    LR  initial     : 0.0
    LR  peak        : 1e-4
    LR  warmup steps: 100   (linear)
    LR  decay steps : 500   (linear, to LR end = 1e-5)
    Activation      : GELU (hidden), tanh (final decoder)
    Negative ELBO   : single MC sample (mc_sim = 1)
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import numpy as np


def lr_schedule(
    step: int,
    peak: float = 1e-4,
    end: float = 1e-5,
    warmup: int = 100,
    decay: int = 500,
) -> float:
    """Linear warmup 0 -> peak then linear decay peak -> end (per addendum)."""
    if step < warmup:
        return peak * (step / max(1, warmup))
    if step < warmup + decay:
        frac = (step - warmup) / decay
        return peak + (end - peak) * frac
    return end


def train_vae(
    cfg: dict,
    images: np.ndarray,
    out_path: str,
) -> None:
    try:
        import torch
        import torch.optim as optim
    except Exception as e:  # pragma: no cover
        print(f"[train_vae] PyTorch unavailable ({e}); writing dummy checkpoint.")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        np.savez(out_path.replace(".pt", ".npz"), latent_dim=cfg["latent_dim"])
        return

    from model.architecture import VAE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(
        c_hid=cfg["c_hid"], latent_dim=cfg["latent_dim"], sigma2=cfg["sigma2"]
    ).to(device)
    opt = optim.Adam(vae.parameters(), lr=cfg["vae_lr_peak"])

    n_epochs = cfg["vae_n_epochs"]
    bs = cfg["vae_batch_size"]
    n_batches = max(1, images.shape[0] // bs)
    step = 0
    for epoch in range(n_epochs):
        idx = np.random.permutation(images.shape[0])
        for k in range(n_batches):
            batch = images[idx[k * bs : (k + 1) * bs]]
            x = torch.as_tensor(batch, dtype=torch.float32, device=device)
            for g in opt.param_groups:
                g["lr"] = lr_schedule(
                    step,
                    peak=cfg["vae_lr_peak"],
                    end=cfg["vae_lr_end"],
                    warmup=cfg["vae_warmup_steps"],
                    decay=cfg["vae_decay_steps"],
                )
            opt.zero_grad()
            loss = vae.neg_elbo(x, mc_sim=cfg["vae_mc_sim"])
            loss.backward()
            opt.step()
            step += 1
        print(
            f"[train_vae] epoch {epoch + 1}/{n_epochs} loss={float(loss):.3f} lr={opt.param_groups[0]['lr']:.2e}"
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": vae.state_dict(),
            "latent_dim": cfg["latent_dim"],
            "c_hid": cfg["c_hid"],
            "sigma2": cfg["sigma2"],
        },
        out_path,
    )
    print(f"[train_vae] checkpoint -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, default="checkpoints/vae.pt")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    import yaml

    with open(args.config) as f:
        cfg_full = yaml.safe_load(f)
    cfg = cfg_full["vae_5_3"]

    from data.loader import load_cifar10, _make_synthetic_cifar10

    images = load_cifar10(train=True, download=True)
    if images is None or args.smoke:
        n = 64 if args.smoke else 256
        print(f"[train_vae] CIFAR-10 unavailable or --smoke; using synthetic n={n}")
        images = _make_synthetic_cifar10(n=n)
        cfg["vae_n_epochs"] = 1
    train_vae(cfg, images, args.out)


if __name__ == "__main__":
    main()
