"""FARE training entrypoint.

Implements:
  Schlarmann, Singh, Croce, Hein. "Robust CLIP: Unsupervised Adversarial
  Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models."
  ICML 2024. https://arxiv.org/abs/2402.12336

The optimization is the unsupervised adversarial fine-tuning objective in
Eq. (3):

    min_theta  E_x [ max_{||z - x||_inf <= eps}
                     || phi_theta(z) - phi_Org(x) ||_2^2 ]

where the inner maximizer is approximated with 10-step PGD (Sec. B.1).
The frozen reference encoder phi_Org is the original OpenAI CLIP ViT-L/14.

Usage:
    python train.py --config configs/default.yaml --output_dir ./checkpoints
                    --epsilon 0.00784   # 2/255 for FARE^2
                    --epsilon 0.01568   # 4/255 for FARE^4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from attacks.pgd import pgd_fare
from data.loader import build_imagenet_loader
from model.architecture import FAREModel
from model.clip_loader import load_clip_vision
from utils.schedule import cosine_schedule_with_warmup


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FARE training")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument(
        "--output_dir", type=str, default=None, help="Override training.output_dir"
    )
    p.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Override adversarial.epsilon (e.g. 0.00784 for 2/255)",
    )
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny smoke training (overrides config to small values)",
    )
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.epsilon is not None:
        cfg["adversarial"]["epsilon"] = float(args.epsilon)
    if args.num_epochs is not None:
        cfg["training"]["num_epochs"] = int(args.num_epochs)
    if args.output_dir is not None:
        cfg["training"]["output_dir"] = args.output_dir
    if args.seed is not None:
        cfg["training"]["seed"] = int(args.seed)
    if args.smoke:
        cfg["smoke"]["enabled"] = True

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Model ----
    print("[FARE] Loading CLIP towers...", flush=True)
    original, finetune, _ = load_clip_vision(
        model_name=cfg["model"]["clip_model_name"],
        pretrained=cfg["model"]["clip_pretrained"],
        device=device,
    )
    model = FAREModel(
        original=original,
        finetune=finetune,
        loss_norm=cfg["loss"]["type"],
    ).to(device)

    # ---- Data ----
    smoke_cfg = cfg.get("smoke", {})
    smoke = smoke_cfg.get("enabled", False)
    bsz_per = (
        smoke_cfg["per_device_batch_size"]
        if smoke
        else cfg["training"]["per_device_batch_size"]
    )
    grad_accum = (
        smoke_cfg["grad_accumulation_steps"]
        if smoke
        else cfg["training"]["grad_accumulation_steps"]
    )
    max_train_steps_smoke = smoke_cfg.get("max_train_steps", None) if smoke else None

    print("[FARE] Building ImageNet loader...", flush=True)
    train_loader = build_imagenet_loader(
        split="train",
        batch_size=bsz_per,
        num_workers=cfg["training"]["num_workers"] if not smoke else 2,
        max_samples=(bsz_per * grad_accum * (max_train_steps_smoke or 1) * 2)
        if smoke
        else None,
    )

    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = steps_per_epoch * cfg["training"]["num_epochs"]
    if max_train_steps_smoke is not None:
        total_steps = min(total_steps, max_train_steps_smoke)
    warmup_steps = max(1, int(round(cfg["schedule"]["warmup_fraction"] * total_steps)))

    # ---- Optimizer ----
    optim = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=cfg["optimizer"]["lr"],
        betas=tuple(cfg["optimizer"]["betas"]),
        eps=cfg["optimizer"]["eps"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    scheduler = cosine_schedule_with_warmup(optim, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"]["precision"] == "fp16")

    eps = cfg["adversarial"]["epsilon"]
    alpha = cfg["adversarial"]["alpha"]
    pgd_steps = cfg["adversarial"]["num_steps"]
    pgd_momentum = cfg["adversarial"]["momentum"]
    pgd_random_start = cfg["adversarial"]["random_start"]

    print(
        f"[FARE] Training: steps={total_steps}, "
        f"eps={eps:.5f} (={int(round(eps * 255))}/255), "
        f"alpha={alpha:.5f}, pgd_steps={pgd_steps}, "
        f"effective_bs={bsz_per * grad_accum}",
        flush=True,
    )

    # ---- Train loop ----
    model.train()
    global_step = 0
    optim.zero_grad(set_to_none=True)
    losses_log = []

    t0 = time.time()
    for epoch in range(cfg["training"]["num_epochs"]):
        for i, (x, _y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)

            # ---- Inner PGD: solve argmax over the eps-ball ----
            x_adv = pgd_fare(
                finetune=model.finetune,
                original=model.original,
                x=x,
                eps=eps,
                alpha=alpha,
                num_steps=pgd_steps,
                momentum=pgd_momentum,
                random_start=pgd_random_start,
            )

            # ---- Outer step: minimize FARE loss ----
            with torch.cuda.amp.autocast(
                enabled=cfg["training"]["precision"] == "fp16"
            ):
                loss = model(x_clean=x, x_adv=x_adv) / grad_accum

            scaler.scale(loss).backward()

            if (i + 1) % grad_accum == 0:
                scaler.step(optim)
                scaler.update()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                losses_log.append(float(loss.item() * grad_accum))
                if global_step % cfg["training"]["log_every"] == 0:
                    elapsed = time.time() - t0
                    avg_loss = sum(losses_log[-50:]) / max(1, len(losses_log[-50:]))
                    lr = optim.param_groups[0]["lr"]
                    print(
                        f"epoch={epoch} step={global_step}/{total_steps} "
                        f"loss={avg_loss:.4f} lr={lr:.2e} "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )

                if max_train_steps_smoke and global_step >= max_train_steps_smoke:
                    break
        if max_train_steps_smoke and global_step >= max_train_steps_smoke:
            break

        # Save per-epoch checkpoint.
        ckpt = output_dir / f"fare_epoch{epoch + 1}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "step": global_step,
                "config": cfg,
                "state_dict": model.finetune.state_dict(),
            },
            ckpt,
        )
        print(f"[FARE] saved {ckpt}", flush=True)

    final_ckpt = output_dir / "fare_final.pt"
    torch.save(
        {
            "step": global_step,
            "config": cfg,
            "state_dict": model.finetune.state_dict(),
        },
        final_ckpt,
    )
    print(f"[FARE] saved {final_ckpt}", flush=True)

    # Dump training log
    with open(output_dir / "train_log.json", "w") as f:
        json.dump(
            {
                "loss_history": losses_log,
                "config": cfg,
                "total_steps": global_step,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
