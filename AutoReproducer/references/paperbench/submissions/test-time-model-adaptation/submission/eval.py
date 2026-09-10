"""eval.py -- run FOA on test-time data and emit metrics.json.

Implements Algorithm 1 of Niu et al., ICML 2024, end-to-end:

    for t in 1..T:
        sample K prompt candidates from N(m^(t), tau^(t)^2 * Sigma^(t))
        evaluate fitness L(p_k) = entropy + lambda * activation discrepancy
        update CMA-ES from {(p_k, v_k)}
        emit prediction with best v_k
        (optional) apply back-to-source activation shifting on e_N^0

Output: writes a JSON report to ``<output_dir>/metrics.json`` with per-
corruption accuracy + ECE and the cross-corruption averages -- the format the
PaperBench judge consumes.

Usage:
    python eval.py --config configs/default.yaml --output_dir /output [--ckpt /output/foa_init.pt]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from data import (
    build_eval_loader,
    build_imagenet_val_loader,
    list_imagenet_c_corruptions,
)
from model import (
    FOA,
    FOAInterval,
    NoAdapt,
    PromptedViT,
    SourceStats,
    T3A,
    TENT,
    build_vit_base,
    quantize_vit_8bit,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--output_dir", type=str, default="/output")
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="optional foa_init.pt produced by train.py",
    )
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="cap each corruption to a small sample count for fast end-to-end testing",
    )
    p.add_argument(
        "--quantize",
        action="store_true",
        help="apply 8-bit weight quantization (Section 4.2)",
    )
    p.add_argument(
        "--baselines",
        action="store_true",
        help="also run NoAdapt / TENT / T3A baselines",
    )
    return p.parse_args()


def _load_or_collect_stats(
    model: PromptedViT, ckpt_path: str | None, Q: int, device: torch.device
) -> SourceStats:
    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"[eval] loading source stats from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        s = ckpt["stats"]
        return SourceStats(
            mu=[t.to(device) for t in s["mu"]],
            sigma=[t.to(device) for t in s["sigma"]],
            mu_final=s["mu_final"].to(device),
        )
    print(f"[eval] no checkpoint -- collecting source stats with Q={Q} samples")
    val_loader = build_imagenet_val_loader(batch_size=min(Q, 16), max_samples=Q)
    return SourceStats.collect(model, val_loader, device=device, max_samples=Q)


def _eval_one_loader(adapter, loader, device) -> dict:
    if isinstance(adapter, FOA):
        return adapter.adapt_and_evaluate(loader, device)
    return adapter.evaluate(loader, device)


def main() -> int:
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    t0 = time.time()

    # --- model ---
    mcfg = cfg.get("model", {})
    n_prompts = int(mcfg.get("num_prompts", 3))
    print(f"[eval] device={device}, N_p={n_prompts}")
    model = build_vit_base(
        num_prompts=n_prompts, pretrained=mcfg.get("pretrained", True)
    )
    if args.quantize or cfg.get("quantization", {}).get("enabled", False):
        bits = int(cfg.get("quantization", {}).get("bits", 8))
        print(f"[eval] applying {bits}-bit weight quantization")
        model = quantize_vit_8bit(model, bits=bits)
    model = model.to(device)

    # --- source stats ---
    stats_cfg = cfg.get("source_stats", {})
    Q = int(stats_cfg.get("num_samples", 32))
    stats = _load_or_collect_stats(model, args.ckpt, Q, device)

    # --- FOA hyperparams ---
    foa_cfg = cfg.get("foa", {})
    eval_cfg = cfg.get("evaluation", {})
    bs = int(eval_cfg.get("batch_size", 64))
    lam_base = float(foa_cfg.get("lambda_disc", 0.4))
    lam = lam_base * (bs / 64.0)
    K = int(foa_cfg.get("popsize", 28))
    use_shift = bool(foa_cfg.get("activation_shift", True))
    print(
        f"[eval] FOA: K={K}, lambda={lam:.3f} (base={lam_base}, BS={bs}), "
        f"act_shift={use_shift}"
    )

    # --- evaluation loop ---
    dataset_name = eval_cfg.get("dataset", "imagenet_c")
    severity = int(eval_cfg.get("severity", 5))
    paths = cfg.get("paths", {})
    cache_per_corruption = int(eval_cfg.get("max_samples_per_corruption", 5000))
    if args.smoke:
        cache_per_corruption = min(cache_per_corruption, 64)

    results: dict = {
        "per_corruption": {},
        "config_summary": {
            "K": K,
            "lambda": lam,
            "BS": bs,
            "severity": severity,
            "activation_shift": use_shift,
            "Q": Q,
            "n_prompts": n_prompts,
            "quantize": bool(args.quantize),
        },
    }

    if dataset_name == "imagenet_c":
        corruptions = eval_cfg.get("corruptions") or list_imagenet_c_corruptions()
        if args.smoke:
            corruptions = corruptions[:3]
        print(
            f"[eval] running over {len(corruptions)} corruptions @ severity {severity}"
        )
        accs, eces = [], []
        for c in corruptions:
            loader = build_eval_loader(
                "imagenet_c",
                root=paths.get("imagenet_c_root", "/data/imagenet-c"),
                corruption=c,
                severity=severity,
                batch_size=bs,
                num_workers=int(eval_cfg.get("num_workers", 2)),
                max_samples=cache_per_corruption,
            )
            # Fresh FOA instance per corruption (paper convention: each
            # corruption is treated as an independent test stream).
            foa = FOA(
                model=model,
                source_stats=stats,
                popsize=K,
                lam=lam,
                activation_shift=use_shift,
                gamma=float(foa_cfg.get("gamma", 1.0)),
                alpha=float(foa_cfg.get("alpha", 0.1)),
                sigma0=float(foa_cfg.get("cma_sigma0", 1.0)),
                seed=int(foa_cfg.get("seed", 2024)),
            )
            metrics = _eval_one_loader(foa, loader, device)
            accs.append(metrics["accuracy"])
            eces.append(metrics.get("ece", 0.0))
            results["per_corruption"][c] = metrics
            print(
                f"[eval]   {c:<22s} acc={metrics['accuracy']:.2f}%  "
                f"ece={metrics.get('ece', 0.0):.2f}%"
            )
        results["average_accuracy"] = float(np.mean(accs)) if accs else 0.0
        results["average_ece"] = float(np.mean(eces)) if eces else 0.0
    else:
        # Generic single-stream eval (ImageNet-R/V2/Sketch/synthetic)
        root = paths.get(f"{dataset_name}_root", None)
        loader = build_eval_loader(
            dataset_name,
            root=root,
            batch_size=bs,
            num_workers=int(eval_cfg.get("num_workers", 2)),
            max_samples=cache_per_corruption,
        )
        foa = FOA(
            model=model,
            source_stats=stats,
            popsize=K,
            lam=lam,
            activation_shift=use_shift,
        )
        metrics = _eval_one_loader(foa, loader, device)
        results["per_corruption"][dataset_name] = metrics
        results["average_accuracy"] = metrics["accuracy"]
        results["average_ece"] = metrics.get("ece", 0.0)
        print(
            f"[eval] {dataset_name}: acc={metrics['accuracy']:.2f}%  "
            f"ece={metrics.get('ece', 0.0):.2f}%"
        )

    # --- optional baselines ---
    if args.baselines or cfg.get("baselines", {}).get("run", False):
        print("[eval] running baselines (NoAdapt, TENT, T3A) on first corruption...")
        first = list(results["per_corruption"].keys())[0]
        loader = build_eval_loader(
            "imagenet_c" if dataset_name == "imagenet_c" else dataset_name,
            root=paths.get("imagenet_c_root", "/data/imagenet-c")
            if dataset_name == "imagenet_c"
            else paths.get(f"{dataset_name}_root", None),
            corruption=first if dataset_name == "imagenet_c" else None,
            severity=severity,
            batch_size=bs,
            max_samples=cache_per_corruption,
        )
        for cls, kwargs in [
            (NoAdapt, {}),
            (TENT, {"lr": float(cfg.get("baselines", {}).get("tent_lr", 1e-3))}),
            (T3A, {"M": int(cfg.get("baselines", {}).get("t3a_M", 100))}),
        ]:
            adp = cls(model, **kwargs) if cls is not NoAdapt else cls(model)
            m = _eval_one_loader(adp, loader, device)
            results.setdefault("baselines", {})[adp.name] = m
            print(f"[eval]   baseline {adp.name}: acc={m['accuracy']:.2f}%")

    results["wallclock_seconds"] = float(time.time() - t0)

    # --- write metrics ---
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] wrote {metrics_path}")
    print(f"[eval] average accuracy = {results['average_accuracy']:.2f}%")
    print(f"[eval] average ECE      = {results['average_ece']:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
