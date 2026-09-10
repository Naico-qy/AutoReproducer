"""Prepare D_PT, D_R, and pairwise (i, j, z_ij) labels.

This script produces the JSON artefacts consumed by `train.py` / `eval.py`.

Outputs:
    /output/d_pt.jsonl       — D_PT (36 P3 tasks × 100 examples)
    /output/d_r_train.jsonl  — 60% of D_R
    /output/d_r_test.jsonl   — 40% of D_R
    /output/pairs_train.jsonl — (i, j, z) triples for training the forecaster
    /output/pairs_test.jsonl  — (i, j, z) triples for evaluation
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# allow running both as `python scripts/prepare_data.py` and as a module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import build_d_pt, load_p3_test, load_mmlu_validation, Example
from data.refinement import split_60_40
from data.tasks import BART0_TEST_TASKS_8


def _to_dict(ex: Example) -> dict:
    return {"uid": ex.uid, "x": ex.x, "y": ex.y, "task": ex.task}


def _write_jsonl(path: str, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/output")
    ap.add_argument("--lm", choices=["bart0", "flan_t5"], default="flan_t5")
    ap.add_argument("--n_per_task", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--smoke", action="store_true", help="Use small sizes for a smoke run"
    )
    args = ap.parse_args()

    if args.smoke:
        args.n_per_task = 10

    # -- D_PT (36 P3 tasks × n_per_task) -----------------------------
    print(f"[prepare_data] Building D_PT with {args.n_per_task}/task ...")
    d_pt = build_d_pt(n_per_task=args.n_per_task, seed=args.seed)
    print(f"[prepare_data] |D_PT| = {len(d_pt)}")
    _write_jsonl(os.path.join(args.out_dir, "d_pt.jsonl"), (_to_dict(e) for e in d_pt))

    # -- D_R (mispredicted on the refinement task) -------------------
    if args.lm == "bart0":
        print("[prepare_data] Loading BART0 P3-Test (8 tasks) ...")
        candidates: list[Example] = []
        for t in BART0_TEST_TASKS_8:
            candidates.extend(load_p3_test(t, max_examples=200 if args.smoke else 1000))
    else:
        print("[prepare_data] Loading MMLU validation ...")
        candidates = load_mmlu_validation(max_per_subject=10 if args.smoke else 50)
    print(f"[prepare_data] candidates: {len(candidates)}")

    # Without an actual base LM available offline we treat *all* candidates as
    # mispredictions (a deliberate proxy when running the smoke pipeline).
    # On a GPU run, eval.py recomputes D_R using the real f_0.
    d_r = candidates

    train, test = split_60_40(d_r, seed=args.seed)
    print(f"[prepare_data] |D_R^Train| = {len(train)} ; |D_R^Test| = {len(test)}")
    _write_jsonl(
        os.path.join(args.out_dir, "d_r_train.jsonl"), (_to_dict(e) for e in train)
    )
    _write_jsonl(
        os.path.join(args.out_dir, "d_r_test.jsonl"), (_to_dict(e) for e in test)
    )

    # -- Pairwise labels (i, j, z) ----------------------------------
    # Without GPU, we sample synthetic z_ij so the rest of the pipeline
    # exercises end-to-end. On a real run, eval.py replaces these.
    import random

    rng = random.Random(args.seed)
    pairs_train = []
    for ex_i in train[: 64 if args.smoke else len(train)]:
        for ex_j in d_pt[: 32 if args.smoke else len(d_pt)]:
            z = int(rng.random() < 0.05)  # ≈5% positive rate (per §4.1)
            pairs_train.append(
                {
                    "uid_i": ex_i.uid,
                    "uid_j": ex_j.uid,
                    "z": z,
                }
            )
    pairs_test = []
    for ex_i in test[: 32 if args.smoke else len(test)]:
        for ex_j in d_pt[: 32 if args.smoke else len(d_pt)]:
            z = int(rng.random() < 0.05)
            pairs_test.append(
                {
                    "uid_i": ex_i.uid,
                    "uid_j": ex_j.uid,
                    "z": z,
                }
            )
    print(
        f"[prepare_data] |pairs_train|={len(pairs_train)} |pairs_test|={len(pairs_test)}"
    )
    _write_jsonl(os.path.join(args.out_dir, "pairs_train.jsonl"), pairs_train)
    _write_jsonl(os.path.join(args.out_dir, "pairs_test.jsonl"), pairs_test)

    print("[prepare_data] DONE.")


if __name__ == "__main__":
    main()
