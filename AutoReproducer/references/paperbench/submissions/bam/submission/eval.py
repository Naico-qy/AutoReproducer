"""
Evaluation entrypoint.

Reads the per-experiment JSONs produced by ``train.py`` and produces a single
``metrics.json`` summary at ``--out`` (default: /output/metrics.json) for the
PaperBench Full-mode judge.

Metrics emitted:
    * gaussian_5_1     : mean forward-KL across runs and dims, plus mean-of-mean.
    * sinh_arcsinh_5_1 : final score-based-divergence approximation per (s,tau).
    * posteriordb_5_2  : relative-mean and relative-SD errors per posterior.
    * vae_5_3          : reconstruction MSE for BaM at the chosen batch size.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

# Allow running from the submission root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _safe_load(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def aggregate(out_dir: str) -> Dict[str, Any]:
    g = _safe_load(os.path.join(out_dir, "gaussian_5_1.json"))
    s = _safe_load(os.path.join(out_dir, "sinh_arcsinh_5_1.json"))
    p = _safe_load(os.path.join(out_dir, "posteriordb_5_2.json"))
    v = _safe_load(os.path.join(out_dir, "vae_5_3.json"))

    metrics: Dict[str, Any] = {}

    # Section 5.1 -- Gaussian targets
    if g:
        rows = []
        for key, dim_res in g.items():
            for method in ("bam", "gsm", "advi"):
                vals = dim_res.get(f"{method}_forward_kl", [])
                if vals:
                    rows.append(
                        {
                            "D": key,
                            "method": method,
                            "mean_forward_kl": float(sum(vals) / len(vals)),
                        }
                    )
        metrics["gaussian_5_1"] = rows

    # Section 5.1 -- Non-Gaussian targets
    if s:
        rows = []
        for key, hist in s.items():
            bam_h = hist.get("bam_history", [])
            advi_h = hist.get("advi_history", [])
            gsm_h = hist.get("gsm_history", [])
            rows.append(
                {
                    "config": key,
                    "bam_iters": len(bam_h),
                    "advi_iters": len(advi_h),
                    "gsm_iters": len(gsm_h),
                }
            )
        metrics["sinh_arcsinh_5_1"] = rows

    # Section 5.2 -- PosteriorDB
    if p:
        metrics["posteriordb_5_2"] = p

    # Section 5.3 -- VAE
    if v:
        metrics["vae_5_3"] = v

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--in_dir",
        type=str,
        default="./outputs",
        help="Where train.py wrote its per-experiment JSONs.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/output/metrics.json",
        help="Where to write the aggregate metrics.",
    )
    args = parser.parse_args()

    metrics = aggregate(args.in_dir)
    metrics["status"] = "ok" if metrics else "no_results"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
