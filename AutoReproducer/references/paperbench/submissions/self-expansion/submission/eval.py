"""SEMA evaluation entry-point.

Re-runs the continual training pipeline (see `train.py`) and computes the
two reported metrics:

  * A_N   : average accuracy at the last step (Eq. 5 in App. B.3).
  * A_bar : average incremental accuracy across all tasks (Eq. 6).

Outputs JSON to `--output` (default /output/metrics.json) so the PaperBench
judge can read the results directly.
"""

from __future__ import annotations

import argparse
import json
import os

from train import load_config, run


def main() -> None:
    p = argparse.ArgumentParser(description="SEMA evaluation")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--output", type=str, default="/output/metrics.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    metrics = run(cfg)

    print(f"A_N    = {metrics['A_N']:.4f}")
    print(f"A_bar  = {metrics['A_bar']:.4f}")

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"metrics -> {args.output}")


if __name__ == "__main__":
    main()
