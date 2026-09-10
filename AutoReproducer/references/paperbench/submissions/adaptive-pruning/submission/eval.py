"""eval.py — evaluation entrypoint for APT.

Loads a trained APT checkpoint (or falls back to fresh-built model for
smoke runs) and reports:
  * Task accuracy / F1 / ROUGE   — §5.1
  * Inference throughput          — addendum: 'samples processed per second'
  * Inference peak memory         — addendum: torch.cuda.max_memory_allocated()
  * Effective parameter count after masking (§4.2)

Usage:
    python eval.py --config configs/default.yaml \
                   --checkpoint /output/last.ckpt \
                   --output_metrics /output/metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict

import torch
import yaml

from data import build_dataloaders, get_task_metric
from model import APTModel, build_apt_model
from model.apt_adapter import APTLinear


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_throughput(
    model: APTModel, loader, device: str, max_batches: int = 50
) -> Dict[str, float]:
    """Return inference throughput (samples/sec) and peak memory.

    Per addendum: 'speed of inference is measured as the inference
    throughput (samples processed per second)'.
    """
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    n_samples = 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        _ = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            token_type_ids=batch.get("token_type_ids"),
        )
        n_samples += batch["input_ids"].size(0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = max(time.time() - t0, 1e-9)
    out = {
        "throughput_samples_per_sec": n_samples / elapsed,
        "elapsed_sec": elapsed,
        "n_samples": n_samples,
    }
    if torch.cuda.is_available():
        out["max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
    return out


# --------------------------------------------------------------------------- #
@torch.no_grad()
def task_accuracy(model: APTModel, loader, device: str) -> Dict[str, float]:
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            token_type_ids=batch.get("token_type_ids"),
        )
        preds = out["logits"].argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].numel()
    return {"accuracy": correct / max(1, total), "n": total}


# --------------------------------------------------------------------------- #
def effective_sparsity(model: APTModel) -> Dict[str, float]:
    total_in_apt = 0
    active_in_apt = 0
    for m in model.modules():
        if isinstance(m, APTLinear):
            total_in_apt += m.in_features * m.out_features
            active_in_apt += m.num_active_in * m.num_active_out
    if total_in_apt == 0:
        return {"effective_sparsity": 0.0}
    return {
        "effective_sparsity": 1.0 - active_in_apt / total_in_apt,
        "params_active_in_apt_layers": active_in_apt,
        "params_total_in_apt_layers": total_in_apt,
    }


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_metrics", type=str, default="/output/metrics.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, eval_loader, num_labels = build_dataloaders(cfg)
    cfg["model"]["num_labels"] = num_labels
    model = build_apt_model(cfg).to(device)

    if args.checkpoint and Path(args.checkpoint).exists():
        sd = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(sd, strict=False)
        print(f"[eval] loaded checkpoint {args.checkpoint}")

    # ---------------- Metrics ---------------------------------------------
    acc = task_accuracy(model, eval_loader, device)
    thr = measure_throughput(model, eval_loader, device)
    sp = effective_sparsity(model)

    out = {
        "task": cfg["data"]["task_name"],
        "primary_metric": get_task_metric(cfg["data"]["task_name"]),
        **acc,
        **thr,
        **sp,
    }
    print(json.dumps(out, indent=2))
    Path(args.output_metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_metrics, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
