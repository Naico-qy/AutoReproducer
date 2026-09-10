"""Linear-probe training entrypoint with the LCA soft-label loss.

This script reproduces the training procedure of §4.3.2 ("Using Class
Taxonomy as Soft Labels") on top of frozen torchvision backbones. Per the
addendum, features M(X) are taken from the layer immediately before the
final FC.  Hyperparameters (Appendix E.5):
    optimizer = AdamW, lr = 0.001, weight_decay = 0.05
    scheduler = cosine + linear warmup (warm-up lr = 1e-5)
    epochs    = 50, batch_size = 1024
    lca_loss: lambda = 0.03, temperature = 25, alignment = CE

Two checkpoints are saved per backbone:
    weights/{backbone}_ce.pt           (cross-entropy only baseline)
    weights/{backbone}_ce_soft.pt      (cross-entropy + LCA soft loss)
The script also writes their weight-space interpolation (Wortsman 2022).

Usage:
    python train.py --config configs/default.yaml --backbone resnet18
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from data.loader import (
    build_dataloader,
    build_imagenet,
    standard_eval_transform,
    standard_train_transform,
)
from model.architecture import LinearProbe, build_vm
from model.lca import (
    WordNetHierarchy,
    KMeansLatentHierarchy,
    build_lca_matrix,
    process_lca_matrix,
    per_class_mean_features,
)
from model.losses import LCAAlignmentLoss, weight_interpolate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LCA-on-the-Line linear-probe training."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        help="torchvision VM name to use as frozen backbone.",
    )
    parser.add_argument(
        "--hierarchy", type=str, default="wordnet", choices=["wordnet", "kmeans"]
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="Run a fast smoke test for grading."
    )
    parser.add_argument("--output-dir", type=str, default="/output")
    return parser.parse_args()


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cosine_with_warmup(
    optimizer, warmup_steps: int, total_steps: int, warmup_lr: float = 1e-5
):
    """Cosine schedule with linear warmup.  Matches paper Appendix E.5."""
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup from warmup_lr -> base_lr
            ratio = step / max(warmup_steps, 1)
            base = base_lrs[0]
            return (warmup_lr + (base - warmup_lr) * ratio) / base
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def extract_features(
    backbone: nn.Module, loader: DataLoader, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Iterate `loader` and return (features, targets) tensors."""
    backbone.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            f = backbone.forward_features(x).detach().cpu()
            feats.append(f)
            labels.append(torch.as_tensor(y))
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def train_one_phase(
    probe: nn.Module,
    feats: torch.Tensor,
    targets: torch.Tensor,
    cfg: dict,
    epochs: int,
    loss_fn: nn.Module,
    device: str,
) -> nn.Module:
    """Train a linear probe for `epochs` epochs over (feats, targets)."""
    probe = probe.to(device)
    bs = int(cfg["linear_probe"]["batch_size"])
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(cfg["linear_probe"]["learning_rate"]),
        weight_decay=float(cfg["linear_probe"]["weight_decay"]),
    )
    n = feats.size(0)
    steps_per_epoch = max(n // bs, 1)
    total_steps = epochs * steps_per_epoch
    warmup = max(steps_per_epoch, 1)
    scheduler = cosine_with_warmup(
        optimizer,
        warmup_steps=warmup,
        total_steps=total_steps,
        warmup_lr=float(cfg["linear_probe"]["warmup_lr"]),
    )
    use_amp = bool(cfg["linear_probe"].get("amp", False)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    feats = feats.to(device)
    targets = targets.to(device)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            idx = perm[s : s + bs]
            xb = feats[idx]
            yb = targets[idx]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = probe(xb)
                loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
    return probe


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    # 1. Backbone (frozen) ----------------------------------------------------
    backbone = build_vm(args.backbone, pretrained=True).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    feat_dim = backbone.feature_dim

    # 2. Build the ImageNet train loader -------------------------------------
    smoke = bool(args.smoke_test or cfg.get("smoke_test", {}).get("enabled", False))
    smoke_samples = int(cfg.get("smoke_test", {}).get("num_eval_samples", 256))
    train_ds = build_imagenet(
        root=cfg["data"]["imagenet_root"],
        split="train",
        image_size=int(cfg["data"]["image_size"]),
        train=True,
        smoke_samples=smoke_samples,
    )
    train_loader = build_dataloader(
        train_ds,
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        shuffle=True,
    )

    # 3. Pre-extract features for fast linear probing ------------------------
    feats, targets = extract_features(backbone, train_loader, device)

    # 4. Build LCA distance matrix ------------------------------------------
    if args.hierarchy == "wordnet":
        wordnet_csv = cfg["data"].get("wordnet_csv", "./resources/imagenet_fiveai.csv")
        if os.path.isfile(wordnet_csv):
            hier = WordNetHierarchy(wordnet_csv)
            lca_raw = build_lca_matrix(
                hier, score="depth"
            )  # depth for soft loss (Appendix D.2)
        else:
            print(
                f"[WARN] WordNet CSV not found at {wordnet_csv}; using zero LCA matrix (soft loss disabled)."
            )
            lca_raw = np.zeros(
                (cfg["data"]["num_classes"], cfg["data"]["num_classes"]),
                dtype=np.float32,
            )
        m_lca = process_lca_matrix(
            lca_raw,
            tree_prefix="WordNet",
            temperature=float(cfg["linear_probe"]["lca_loss"]["temperature"]),
        )
    else:
        # K-means latent hierarchy from per-class mean features.
        class_means = per_class_mean_features(
            feats, targets, num_classes=cfg["data"]["num_classes"]
        ).numpy()
        klh = KMeansLatentHierarchy(num_levels=int(cfg["lca"]["kmeans_levels"])).fit(
            class_means
        )
        lca_raw = klh.matrix()
        m_lca = process_lca_matrix(
            lca_raw,
            tree_prefix="kmeans",
            temperature=float(cfg["linear_probe"]["lca_loss"]["temperature"]),
        )

    # 5. Phase A: CE-only baseline ------------------------------------------
    epochs = (
        cfg["smoke_test"]["linear_probe_epochs"]
        if smoke
        else int(cfg["linear_probe"]["epochs"])
    )
    probe_ce = LinearProbe(feat_dim, num_classes=cfg["data"]["num_classes"])
    probe_ce = train_one_phase(
        probe_ce,
        feats,
        targets,
        cfg,
        epochs,
        loss_fn=nn.CrossEntropyLoss(),
        device=device,
    )
    torch.save(probe_ce.state_dict(), weights_dir / f"{args.backbone}_ce.pt")

    # 6. Phase B: CE + LCA soft loss ----------------------------------------
    probe_soft = LinearProbe(feat_dim, num_classes=cfg["data"]["num_classes"])
    soft_loss = LCAAlignmentLoss(
        lca_matrix=m_lca.to(device),
        alignment_mode=cfg["linear_probe"]["lca_loss"]["alignment_mode"],
        lambda_weight=float(cfg["linear_probe"]["lca_loss"]["lambda_weight"]),
    )
    probe_soft = train_one_phase(
        probe_soft, feats, targets, cfg, epochs, loss_fn=soft_loss, device=device
    )
    torch.save(probe_soft.state_dict(), weights_dir / f"{args.backbone}_ce_soft.pt")

    # 7. Weight-space interpolation (Wortsman 2022) -------------------------
    alpha = float(cfg["linear_probe"]["lca_loss"]["weight_interpolation_alpha"])
    interp = weight_interpolate(
        probe_ce.state_dict(), probe_soft.state_dict(), alpha=alpha
    )
    torch.save(interp, weights_dir / f"{args.backbone}_ce_soft_interp_a{alpha}.pt")

    # 8. Persist the LCA matrix for re-use during evaluation.
    torch.save(m_lca.cpu(), weights_dir / f"{args.backbone}_M_LCA.pt")

    summary = {
        "backbone": args.backbone,
        "hierarchy": args.hierarchy,
        "epochs": epochs,
        "feature_dim": feat_dim,
        "num_train_samples": int(feats.size(0)),
        "lca_matrix_shape": list(m_lca.shape),
        "smoke_test": smoke,
    }
    with open(output_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
