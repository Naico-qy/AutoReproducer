"""train.py -- main entrypoint for FOA test-time adaptation.

NOTE: FOA is an *online* test-time adaptation method that does not require
offline training.  This script therefore performs the only "training-like"
step the paper specifies:

    1. Load the ImageNet-1K-pretrained ViT-Base from `timm`.
    2. Collect source in-distribution statistics {mu_i^S, sigma_i^S} from
       Q=32 ImageNet-1K validation samples (Section 3.1; Figure 2(c)).
    3. Save a checkpoint that bundles (a) the ID statistics and (b) the
       initial prompt + CMA-ES seed for downstream eval.

For the actual TTA loop (which IS the optimization in this paper), see
`eval.py` -- following the paper's Algorithm 1, the prompt is learned
online over the test stream, not offline.

Usage:
    python train.py --config configs/default.yaml --output_dir /output

Reference: Niu et al., "Test-Time Model Adaptation with Only Forward
Passes", ICML 2024, Algorithm 1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

from data import build_imagenet_val_loader
from model import build_vit_base
from model.foa import SourceStats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--output_dir", type=str, default="/output")
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="override Q (number of ID samples for stats)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"[train] config not found: {cfg_path}", file=sys.stderr)
        return 2
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # 1. Build model
    model_cfg = cfg.get("model", {})
    n_prompts = int(model_cfg.get("num_prompts", 3))
    pretrained = bool(model_cfg.get("pretrained", True))
    print(f"[train] building ViT-Base with N_p={n_prompts}, pretrained={pretrained}")
    model = build_vit_base(num_prompts=n_prompts, pretrained=pretrained).to(device)

    # 2. Collect source stats
    stats_cfg = cfg.get("source_stats", {})
    Q = int(args.num_samples or stats_cfg.get("num_samples", 32))
    print(f"[train] collecting source ID statistics from Q={Q} ImageNet val samples...")
    val_loader = build_imagenet_val_loader(batch_size=min(Q, 16), max_samples=Q)
    stats = SourceStats.collect(model, val_loader, device=device, max_samples=Q)
    n_layers = len(stats.mu)
    print(
        f"[train] collected stats for {n_layers} layers, mu_final shape = {tuple(stats.mu_final.shape)}"
    )

    # 3. Save bundle
    ckpt = {
        "config": cfg,
        "num_prompts": n_prompts,
        "embed_dim": model.embed_dim,
        "stats": {
            "mu": [t.cpu() for t in stats.mu],
            "sigma": [t.cpu() for t in stats.sigma],
            "mu_final": stats.mu_final.cpu(),
        },
        "initial_prompt": model.prompt.detach().cpu(),
    }
    ckpt_path = out_dir / "foa_init.pt"
    torch.save(ckpt, ckpt_path)
    print(f"[train] wrote {ckpt_path}")

    summary = {
        "model": model_cfg.get("name", "vit_base_patch16_224"),
        "num_prompts": n_prompts,
        "Q": Q,
        "n_layers": n_layers,
        "checkpoint": str(ckpt_path),
    }
    with (out_dir / "train_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("[train] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
