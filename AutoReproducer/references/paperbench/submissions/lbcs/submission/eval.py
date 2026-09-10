"""LBCS — evaluation entrypoint.

Loads a saved coreset mask + target model and reports test-accuracy + coreset
size. Optionally also retrains a fresh target model from the saved mask.

Usage:
    python eval.py --config configs/default.yaml --output-dir ./out [--retrain]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import yaml

from data.loader import get_dataset
from model.architecture import build_model
from utils.inner import compute_accuracy, inner_train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument(
        "--retrain",
        action="store_true",
        help="retrain a fresh target network on the saved mask",
    )
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["experiment"]["output_dir"] = args.output_dir
    out_dir = cfg["experiment"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set, test_set, num_classes, in_channels = get_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["data_root"],
        download=cfg["dataset"]["download"],
    )

    mask_path = os.path.join(out_dir, "coreset_mask.npy")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"no mask at {mask_path} — run train.py first")
    mask = np.load(mask_path).astype(np.uint8)
    print(f"[lbcs.eval] mask size = {int(mask.sum())} / {mask.shape[0]}")

    # Re-train if requested OR no checkpoint present
    ckpt_path = os.path.join(out_dir, "target_model.pt")
    target_arch = cfg["coreset_train"]["target_arch"]
    net = build_model(target_arch, num_classes=num_classes, in_channels=in_channels)
    if args.retrain or not os.path.exists(ckpt_path):
        net = inner_train(
            net,
            train_set,
            mask,
            epochs=int(cfg["coreset_train"]["epochs"]),
            batch_size=int(cfg["coreset_train"]["batch_size"]),
            lr=float(cfg["coreset_train"]["lr"]),
            optimizer=cfg["coreset_train"]["optimizer"],
            momentum=float(cfg["coreset_train"]["momentum"]),
            weight_decay=float(cfg["coreset_train"]["weight_decay"]),
            device=device,
            scheduler=cfg["coreset_train"].get("scheduler", "none"),
        )
    else:
        ck = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(ck["state_dict"])

    acc = compute_accuracy(net, test_set, device=device)
    print(f"[lbcs.eval] test_acc = {acc * 100:.2f}%, coreset_size = {int(mask.sum())}")

    # Update / write metrics.json
    metrics_path = os.path.join(out_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}
    metrics.update(
        {
            "test_accuracy_eval": float(acc),
            "coreset_size_eval": int(mask.sum()),
        }
    )
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
