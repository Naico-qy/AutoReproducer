"""DPMs-ANT main training entrypoint -- implements Algorithm 1.

  Wang et al., "Bridging Data Gaps in Diffusion Models with Adversarial
  Noise-Based Transfer Learning", ICML 2024.

Pipeline:
   1) Load config (configs/default.yaml or a target-specific override)
   2) Build the pre-trained U-Net ε_θ
   3) Build the binary classifier p_phi(y | x_t)  (frozen here; train via
      scripts/train_classifier.py before running this)
   4) Wrap U-Net in AdaptedUNet so only the adaptor ψ is trainable
   5) For each iteration of Algorithm 1:
        - sample x0 ~ q(x0)  (10-shot target dataset)
        - sample t ~ Uniform(1..T)
        - obtain ε* via Eq. (7) PGD ascent on ε
        - update ψ to minimize Eq. (8)  L(ψ)
   6) Save the trained adaptor and a few generated samples.

Usage:
   python train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_config, set_seed, EMA, save_metrics
from data.loader import build_target_loader
from model.unet import UNet
from model.classifier import BinaryNoiseClassifier
from model.architecture import DPMsANT, DPMsANTConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPMs-ANT training (Algorithm 1)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--source-ckpt", default=None, help="path to pre-trained ε_θ checkpoint"
    )
    p.add_argument(
        "--classifier-ckpt",
        default=None,
        help="path to fine-tuned binary classifier checkpoint",
    )
    p.add_argument("--out", default=None, help="output dir override")
    p.add_argument(
        "--smoke", action="store_true", help="short run for CI / smoke testing"
    )
    return p.parse_args()


def build_unet(cfg: dict, image_size: int) -> UNet:
    u = cfg["unet"]
    return UNet(
        in_channels=u["in_channels"],
        out_channels=u["out_channels"],
        base_channels=u["base_channels"],
        channel_mult=tuple(u["channel_mult"]),
        num_res_blocks=u["num_res_blocks"],
        attention_resolutions=tuple(u["attention_resolutions"]),
        image_size=image_size,
        dropout=u["dropout"],
        num_heads=u["num_heads"],
    )


def build_classifier(cfg: dict, image_size: int) -> BinaryNoiseClassifier:
    return BinaryNoiseClassifier(
        in_channels=cfg["image_channels"],
        image_size=image_size,
        num_classes=cfg["classifier"]["num_classes"],
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = torch.device(
        cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    out_dir = args.out or cfg.get("output_dir", "./outputs")
    os.makedirs(out_dir, exist_ok=True)

    image_size = cfg["image_size"]
    if cfg["framework"] == "ldm":
        # LDM operates on a 4-channel z-latent at 64x64 (Rombach et al. 2022)
        image_size = 64
        cfg["image_channels"] = 4
        cfg["unet"]["in_channels"] = 4
        cfg["unet"]["out_channels"] = 4

    # -----------------------------------------------------------------
    # 1) Build models
    # -----------------------------------------------------------------
    unet = build_unet(cfg, image_size)
    if args.source_ckpt and os.path.isfile(args.source_ckpt):
        sd = torch.load(args.source_ckpt, map_location="cpu")
        sd = sd.get("state_dict", sd)
        unet.load_state_dict(sd, strict=False)
        print(f"[train] loaded source U-Net from {args.source_ckpt}")
    else:
        print(
            "[train] WARNING: no --source-ckpt; running with randomly-initialized U-Net "
            "(smoke-test mode; real reproduction requires the pre-trained DDPM/LDM weights)"
        )

    classifier = build_classifier(cfg, image_size)
    if args.classifier_ckpt and os.path.isfile(args.classifier_ckpt):
        sd = torch.load(args.classifier_ckpt, map_location="cpu")
        sd = sd.get("state_dict", sd)
        classifier.load_state_dict(sd, strict=False)
        print(f"[train] loaded classifier from {args.classifier_ckpt}")
    else:
        print(
            "[train] WARNING: no --classifier-ckpt; classifier guidance term will be "
            "approximate (smoke-test mode)"
        )

    # -----------------------------------------------------------------
    # 2) Wrap in DPMsANT
    # -----------------------------------------------------------------
    dpm_cfg = DPMsANTConfig(
        T=cfg["diffusion"]["num_timesteps"],
        beta_schedule=cfg["diffusion"]["beta_schedule"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
        adaptor_c=cfg["adaptor"]["c"],
        adaptor_d=cfg["adaptor"]["d"],
        adaptor_heads=cfg["adaptor"]["attention_heads"],
        adaptor_zero_init=cfg["adaptor"]["zero_init"],
        freeze_backbone=cfg["train"]["freeze_backbone"],
        gamma=cfg["similarity"]["gamma"],
        use_adv_noise=cfg["adversarial"]["enabled"],
        adv_J=cfg["adversarial"]["J"],
        adv_omega=cfg["adversarial"]["omega"],
    )
    model = DPMsANT(unet=unet, classifier=classifier, cfg=dpm_cfg, device=device)
    n_train = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in unet.parameters()) + n_train
    print(
        f"[train] trainable params (ψ) = {n_train:,} | total = {n_total:,} | "
        f"rate = {n_train / max(n_total, 1) * 100:.2f}% (paper reports ~1.3% / 1.6%)"
    )

    # -----------------------------------------------------------------
    # 3) Optimizer + EMA
    # -----------------------------------------------------------------
    train_cfg = cfg["train"]
    params = list(model.trainable_parameters())
    if train_cfg["optimizer"].lower() == "adamw":
        opt = torch.optim.AdamW(
            params, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
        )
    else:
        opt = torch.optim.Adam(params, lr=train_cfg["lr"])
    ema = EMA(params, decay=train_cfg["ema_decay"])

    # -----------------------------------------------------------------
    # 4) Data loader
    # -----------------------------------------------------------------
    target_loader = build_target_loader(
        root=cfg["data"]["target_root"],
        batch_size=train_cfg["batch_size"],
        image_size=image_size,
        num_images=cfg["data"]["num_target_images"],
        num_workers=cfg["data"]["num_workers"],
        augment=cfg["data"]["augment"],
    )

    # -----------------------------------------------------------------
    # 5) Algorithm 1 main loop
    # -----------------------------------------------------------------
    iters = train_cfg["iterations"]
    if args.smoke:
        iters = min(iters, 2)
    log_every = train_cfg["log_every"]
    losses = []
    t0 = time.time()
    step = 0
    while step < iters:
        for x0 in target_loader:
            x0 = x0.to(device, non_blocking=True)
            loss = model.training_step(x0)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, train_cfg["grad_clip"])
            opt.step()
            ema.update(params)

            losses.append(float(loss.detach()))
            if step % log_every == 0:
                rate = (step + 1) / max(time.time() - t0, 1e-6)
                print(
                    f"[train] iter {step:04d}/{iters} loss={loss.item():.4f}  "
                    f"({rate:.2f} it/s)",
                    flush=True,
                )
            step += 1
            if step >= iters:
                break

    # -----------------------------------------------------------------
    # 6) Save adaptor + metrics
    # -----------------------------------------------------------------
    ema.copy_to(params)
    ckpt_path = os.path.join(out_dir, "adaptor.pt")
    torch.save(
        {
            "down_adaptors": model.eps_model.down_adaptors.state_dict(),
            "mid_adaptor": model.eps_model.mid_adaptor.state_dict(),
            "up_adaptors": model.eps_model.up_adaptors.state_dict(),
            "config": cfg,
        },
        ckpt_path,
    )
    print(f"[train] saved adaptor -> {ckpt_path}", flush=True)

    save_metrics(
        {
            "framework": cfg["framework"],
            "iterations": iters,
            "final_loss": losses[-1] if losses else None,
            "mean_loss_last_10": (sum(losses[-10:]) / max(len(losses[-10:]), 1))
            if losses
            else None,
            "trainable_params": n_train,
            "total_params": n_total,
            "parameter_rate_percent": (n_train / max(n_total, 1)) * 100.0,
            "wall_time_sec": time.time() - t0,
        },
        os.path.join(out_dir, "train_metrics.json"),
    )


if __name__ == "__main__":
    main()
