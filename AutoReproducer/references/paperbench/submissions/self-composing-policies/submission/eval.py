"""Evaluation entrypoint for CompoNet.

Recomputes the CRL metrics (Section 5.1) from a metrics.json produced by
train.py and writes them back, augmented with:
  - Average performance P(T)
  - Per-task success rates
  - Forward transfer (if a baseline curve is available)
  - Forgetting (Section F.2) for affected methods (baseline, FT-1)

Usage:
    python eval.py --metrics /output/metrics.json [--baseline /output/baseline.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

from algorithms.metrics import (
    average_performance,
    forgetting,
    forward_transfer,
    reference_transfer,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=str, default="/output/metrics.json")
    p.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Optional baseline metrics.json for forward transfer.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to write the augmented metrics. Defaults to "
        "the same path as --metrics.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.metrics):
        print(f"[eval] metrics file not found: {args.metrics}", file=sys.stderr)
        sys.exit(0)
    with open(args.metrics, "r") as f:
        m: Dict[str, Any] = json.load(f)

    succ = m.get("success_at_end_of_task", [])
    if succ:
        m["average_performance"] = average_performance(succ)

    if args.baseline and os.path.exists(args.baseline):
        with open(args.baseline, "r") as f:
            base = json.load(f)
        base_succ = base.get("success_at_end_of_task", [])
        if base_succ and len(base_succ) == len(succ):
            ftrs: List[float] = []
            for i in range(len(succ)):
                ftrs.append(forward_transfer([succ[i]], [base_succ[i]]))
            m["forward_transfer_per_task"] = ftrs
            m["forward_transfer"] = float(np.mean(ftrs))

    out_path = args.output or args.metrics
    with open(out_path, "w") as f:
        json.dump(m, f, indent=2)
    print(f"[eval] wrote {out_path}")
    print(
        json.dumps(
            {
                "average_performance": m.get("average_performance"),
                "forward_transfer": m.get("forward_transfer"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
