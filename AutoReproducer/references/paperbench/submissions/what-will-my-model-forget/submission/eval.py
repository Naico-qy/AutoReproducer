"""Evaluate forecasters and end-to-end model refinement.

Writes the final metrics dictionary to `output.metrics_path`
(default `/output/metrics.json`).

Usage:
    python eval.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forecasting.eval_loop import f1_score


def _load_jsonl(path: str) -> list[dict]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _idx(items: list[dict]) -> dict[str, dict]:
    return {it["uid"]: it for it in items}


def evaluate_threshold(ckpt_dir: str, pairs_test: list[dict]) -> dict[str, float]:
    """Predict using the saved γ + counts; return F1 / P / R."""
    path = os.path.join(ckpt_dir, "threshold.json")
    if not os.path.exists(path):
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "skipped": True}
    state = json.load(open(path))
    gamma = state["gamma"]
    counts = state["counts"]
    preds, labels = [], []
    for p in pairs_test:
        preds.append(int(counts.get(p["uid_j"], 0) >= gamma))
        labels.append(int(p["z"]))
    f1, prec, rec = f1_score(preds, labels)
    return {"f1": f1 * 100.0, "precision": prec * 100.0, "recall": rec * 100.0}


def evaluate_repr(ckpt_dir: str, pairs_test, idx_i, idx_j) -> dict[str, float]:
    """Run the trained representation forecaster on the test pairs."""
    path = os.path.join(ckpt_dir, "representation.pt")
    if not os.path.exists(path):
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "skipped": True}
    try:
        import torch
        from model.repr_forecaster import (
            RepresentationForecaster,
            ReprForecasterConfig,
        )
        from model.encoder import EncoderConfig

        device = "cuda" if torch.cuda.is_available() else "cpu"
        fc = RepresentationForecaster(ReprForecasterConfig(), EncoderConfig()).to(
            device
        )
        fc.load_state_dict(torch.load(path, map_location=device))
        fc.eval()
        preds, labels = [], []
        with torch.no_grad():
            for p in pairs_test:
                ex_i = idx_i.get(p["uid_i"])
                ex_j = idx_j.get(p["uid_j"])
                if ex_i is None or ex_j is None:
                    continue
                s = fc(ex_i["x"], ex_i["y"], ex_j["x"], ex_j["y"], uid_j=ex_j["uid"])
                preds.append(int(torch.sigmoid(s).item() > 0.5))
                labels.append(int(p["z"]))
        f1, prec, rec = f1_score(preds, labels)
        return {"f1": f1 * 100.0, "precision": prec * 100.0, "recall": rec * 100.0}
    except Exception as e:  # pragma: no cover
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data_dir", default="/output")
    ap.add_argument("--ckpt_dir", default="/output/logs")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pairs_test = _load_jsonl(os.path.join(args.data_dir, "pairs_test.jsonl"))
    d_r_train = _load_jsonl(os.path.join(args.data_dir, "d_r_train.jsonl"))
    d_r_test = _load_jsonl(os.path.join(args.data_dir, "d_r_test.jsonl"))
    d_pt = _load_jsonl(os.path.join(args.data_dir, "d_pt.jsonl"))

    idx_i = _idx(d_r_train + d_r_test)
    idx_j = _idx(d_pt)

    metrics = {
        "paper": "What Will My Model Forget? (Jin & Ren, ICML 2024)",
        "config_default_lm": cfg["default_lm"],
        "n_d_pt": len(d_pt),
        "n_d_r_train": len(d_r_train),
        "n_d_r_test": len(d_r_test),
        "n_pairs_test": len(pairs_test),
        "forecasting_f1": {
            "threshold": evaluate_threshold(args.ckpt_dir, pairs_test),
            "representation": evaluate_repr(args.ckpt_dir, pairs_test, idx_i, idx_j),
        },
    }

    metrics_path = cfg["output"]["metrics_path"]
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    json.dump(metrics, open(metrics_path, "w"), indent=2)
    print(f"[eval] metrics written to {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
