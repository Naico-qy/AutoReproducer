"""FARE evaluation entrypoint.

Implements the zero-shot classification evaluation of Sec. 4.3 / Table 4 of
the paper. For each dataset:
  1. Build a CLIP zero-shot classifier (text-prompted templates).
  2. Compute clean accuracy on all samples.
  3. Compute robust accuracy under l_inf APGD-CE + APGD-DLR-Targeted on
     1000 sampled images at eps in {2/255, 4/255}, mirroring Sec. B.10.

Outputs metrics JSON to `--metrics_path` (default /output/metrics.json) so
the PaperBench Reproduction judge can pick it up.

Usage:
    python eval.py --config configs/default.yaml \
                   --checkpoint ./checkpoints/fare_final.pt \
                   --metrics_path /output/metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import yaml

from attacks.apgd import apgd_ensemble_eval
from data.loader import build_zero_shot_loader
from model.architecture import FAREModel
from model.clip_loader import load_clip_vision
from utils.zeroshot import (
    DEFAULT_TEMPLATES,
    ZeroShotClassifier,
    build_zeroshot_classifier,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FARE evaluation")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to FARE finetune state_dict (if None, evaluates original CLIP).",
    )
    p.add_argument("--metrics_path", type=str, default="/output/metrics.json")
    p.add_argument(
        "--datasets", nargs="*", default=None, help="Override list of datasets"
    )
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def clean_accuracy(
    classifier: ZeroShotClassifier, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    logits = classifier(x)
    return (logits.argmax(dim=1) == y).float()


def robust_accuracy(
    classifier: ZeroShotClassifier, x, y, eps: float, n_iter: int
) -> torch.Tensor:
    x_adv = apgd_ensemble_eval(
        classifier,
        x,
        y,
        eps=eps,
        n_iter=n_iter,
        num_classes=classifier.text_weights.shape[1],
    )
    with torch.no_grad():
        return (classifier(x_adv).argmax(dim=1) == y).float()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    smoke = args.smoke or cfg.get("smoke", {}).get("enabled", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load encoders ----
    print("[FARE-eval] Loading CLIP towers...", flush=True)
    original, finetune, _ = load_clip_vision(
        model_name=cfg["model"]["clip_model_name"],
        pretrained=cfg["model"]["clip_pretrained"],
        device=device,
    )

    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        sd = ckpt.get("state_dict", ckpt)
        missing, unexpected = finetune.load_state_dict(sd, strict=False)
        print(
            f"[FARE-eval] loaded checkpoint {args.checkpoint} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})",
            flush=True,
        )
    else:
        print(
            "[FARE-eval] WARNING: no checkpoint loaded — evaluating fresh CLIP weights",
            flush=True,
        )

    # We need text encoder too — re-load via open_clip to get the tokenizer
    # and text encoder.
    import open_clip  # type: ignore

    full_model, _, _ = open_clip.create_model_and_transforms(
        cfg["model"]["clip_model_name"], pretrained=cfg["model"]["clip_pretrained"]
    )
    full_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(cfg["model"]["clip_model_name"])

    finetune.eval()

    # ---- Iterate datasets ----
    datasets = args.datasets or cfg["evaluation"]["datasets"]
    radii = cfg["evaluation"]["attack_radii"]
    n_iter = cfg["evaluation"]["apgd_iters"]
    max_eval = (
        cfg["smoke"]["max_eval_samples"]
        if smoke
        else (args.max_eval_samples or cfg["evaluation"]["num_eval_samples"])
    )

    metrics: Dict[str, Dict[str, float]] = {}
    for ds_name in datasets:
        try:
            print(f"\n[FARE-eval] === {ds_name} ===", flush=True)
            loader, classnames = build_zero_shot_loader(
                ds_name,
                batch_size=8 if smoke else 32,
                max_samples=max_eval,
                resolution=cfg["model"]["image_resolution"],
            )
            text_w = build_zeroshot_classifier(
                full_model.encode_text,
                tokenizer,
                classnames,
                templates=DEFAULT_TEMPLATES,
                device=device,
            )
            classifier = (
                ZeroShotClassifier(
                    vision_encoder=finetune,
                    text_weights=text_w,
                    logit_scale=100.0,
                )
                .to(device)
                .eval()
            )

            clean_acc_acc, n = 0.0, 0
            adv_acc = {f"eps={r:.5f}": 0.0 for r in radii}

            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                clean_acc_acc += clean_accuracy(classifier, x, y).sum().item()
                for r in radii:
                    adv_acc[f"eps={r:.5f}"] += (
                        robust_accuracy(
                            classifier,
                            x,
                            y,
                            eps=r,
                            n_iter=n_iter,
                        )
                        .sum()
                        .item()
                    )
                n += x.shape[0]
                if max_eval and n >= max_eval:
                    break

            metrics[ds_name] = {
                "clean_acc": clean_acc_acc / max(1, n),
                "n_samples": n,
            }
            for r in radii:
                metrics[ds_name][f"adv_acc@{int(round(r * 255))}/255"] = adv_acc[
                    f"eps={r:.5f}"
                ] / max(1, n)

            print(f"[FARE-eval] {ds_name}: {metrics[ds_name]}", flush=True)

        except Exception as e:  # noqa: BLE001
            metrics[ds_name] = {"error": str(e)}
            print(f"[FARE-eval] {ds_name} skipped: {e}", flush=True)

    # ---- Aggregate ----
    valid_metrics = {k: v for k, v in metrics.items() if "error" not in v}
    if valid_metrics:
        avg_clean = sum(v["clean_acc"] for v in valid_metrics.values()) / len(
            valid_metrics
        )
        metrics["__avg__"] = {"clean_acc": avg_clean}
        for r in radii:
            key = f"adv_acc@{int(round(r * 255))}/255"
            avg = sum(v.get(key, 0.0) for v in valid_metrics.values()) / len(
                valid_metrics
            )
            metrics["__avg__"][key] = avg

    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[FARE-eval] wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
