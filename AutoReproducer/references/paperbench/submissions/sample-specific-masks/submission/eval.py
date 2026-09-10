"""
SMM evaluation entrypoint.

Loads a trained checkpoint and reports top-1 accuracy on the target test set.
Also prints the average accuracy summary the paper reports in Tables 1, 2,
and 10 (when invoked with multiple --dataset args via reproduce.sh).

Usage:
    python eval.py --config configs/default.yaml --ckpt output/checkpoints/best.pt
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from data.loader import build_dataloaders
from model.architecture import SMM
from model.label_mapping import LabelMapping


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--ckpt", type=str, default="output/checkpoints/best.pt")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--network", type=str, default=None)
    p.add_argument("--method", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def evaluate(
    model: SMM, label_map: LabelMapping, loader, device: torch.device
) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits_p = model(x)
        logits_t = label_map(logits_p)
        pred = logits_t.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for k, v in vars(args).items():
        if v is not None and k not in ("config", "ckpt"):
            cfg[k] = v

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    bs = int(cfg["batch_size"])
    if cfg["dataset"].lower() in ("dtd", "oxfordpets"):
        bs = int(cfg.get("batch_size_small", 64))
    _, test_loader, num_classes = build_dataloaders(
        name=cfg["dataset"],
        root=cfg["data_root"],
        network=cfg["network"],
        batch_size=bs,
        num_workers=int(cfg.get("num_workers", 4)),
        seed=int(cfg["seed"]),
    )

    # Model
    model = SMM(
        network=cfg["network"],
        method=cfg["method"],
        num_pool_layers=int(cfg["mask_num_pool_layers"]),
    ).to(device)
    label_map = LabelMapping(num_target=num_classes).to(device)

    ckpt_path = Path(args.ckpt)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.delta.data.copy_(ckpt["delta"].to(device))
        if model.fmask is not None and ckpt.get("fmask") is not None:
            model.fmask.load_state_dict(ckpt["fmask"])
        label_map.y_sub = ckpt["y_sub"].to(device)
        print(f"[eval] loaded checkpoint {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
    else:
        print(f"[eval] WARNING: checkpoint {ckpt_path} not found -- using random init")

    acc = evaluate(model, label_map, test_loader, device)
    print(
        f"[eval] {cfg['dataset']} / {cfg['network']} / {cfg['method']} -> "
        f"top-1 accuracy = {acc * 100:.2f}%"
    )

    out = {
        "dataset": cfg["dataset"],
        "network": cfg["network"],
        "method": cfg["method"],
        "mapping_method": cfg["mapping_method"],
        "test_accuracy": acc,
    }
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
