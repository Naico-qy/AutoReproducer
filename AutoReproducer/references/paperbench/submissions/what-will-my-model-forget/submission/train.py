"""Train a forecasting model.

Usage:

    python train.py --config configs/default.yaml --method threshold
    python train.py --config configs/default.yaml --method logit
    python train.py --config configs/default.yaml --method representation

Outputs are written under the directory specified by `output.log_dir` in the
config (default `/output/logs`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

# allow running both as `python train.py` and as a module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forecasting.train_loop import (
    train_threshold,
    train_logit,
    train_repr,
)


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


def _build_index(items: list[dict]) -> dict[str, dict]:
    return {it["uid"]: it for it in items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--method", choices=["threshold", "logit", "representation"], required=True
    )
    ap.add_argument(
        "--data_dir", default="/output", help="Where prepare_data.py wrote artefacts"
    )
    ap.add_argument("--out_dir", default="/output/logs")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    os.makedirs(args.out_dir, exist_ok=True)

    # ----- load artefacts ----------------------------------------------
    d_r_train = _load_jsonl(os.path.join(args.data_dir, "d_r_train.jsonl"))
    d_pt = _load_jsonl(os.path.join(args.data_dir, "d_pt.jsonl"))
    pairs_train = _load_jsonl(os.path.join(args.data_dir, "pairs_train.jsonl"))

    idx_i = _build_index(d_r_train)
    idx_j = _build_index(d_pt)

    # ----- threshold (§3.1) --------------------------------------------
    if args.method == "threshold":
        flat = [(p["uid_i"], p["uid_j"], int(p["z"])) for p in pairs_train]
        fc = train_threshold(
            flat,
            min_g=cfg["forecasting"]["threshold"]["gamma_search_min"],
            max_g=cfg["forecasting"]["threshold"]["gamma_search_max"],
        )
        ckpt = os.path.join(args.out_dir, "threshold.json")
        json.dump({"gamma": fc.gamma, "counts": fc.counts}, open(ckpt, "w"))
        print(f"[train] saved {ckpt} (γ={fc.gamma})")
        return

    # The next two methods need PyTorch; import lazily.
    try:
        import torch
    except Exception:
        print("[train] PyTorch not available — skipping {} (smoke)".format(args.method))
        json.dump(
            {"skipped": True, "reason": "no_pytorch"},
            open(os.path.join(args.out_dir, f"{args.method}.json"), "w"),
        )
        return

    # ----- representation (§3.3) ---------------------------------------
    if args.method == "representation":
        train_pairs_full = []
        for p in pairs_train:
            ex_i = idx_i.get(p["uid_i"])
            ex_j = idx_j.get(p["uid_j"])
            if ex_i is None or ex_j is None:
                continue
            train_pairs_full.append((ex_i, ex_j, int(p["z"])))
        if not train_pairs_full:
            print("[train] no pairs — skipping representation")
            return
        fc = train_repr(
            train_pairs_full,
            encoder_name=cfg["forecasting"]["encoder"]["name"],
            use_frequency_prior=cfg["forecasting"]["representation_based"][
                "use_frequency_prior"
            ],
            epochs=cfg["forecasting"]["representation_based"]["epochs"],
            lr=cfg["forecasting"]["representation_based"]["lr"],
        )
        torch.save(fc.state_dict(), os.path.join(args.out_dir, "representation.pt"))
        print("[train] saved representation.pt")
        return

    # ----- logit (§3.2) ------------------------------------------------
    # Building the f_0 / Δ caches requires running the base PTLM, which is
    # only feasible on a GPU. We document that here and short-circuit when
    # those caches don't exist.
    cache_path = os.path.join(args.data_dir, "logit_caches.pt")
    if not os.path.exists(cache_path):
        print(
            "[train] logit caches not found — skipping logit forecaster "
            "(requires GPU run; see eval.py)."
        )
        json.dump(
            {"skipped": True, "reason": "no_gpu_caches"},
            open(os.path.join(args.out_dir, "logit.json"), "w"),
        )
        return

    caches = torch.load(cache_path, map_location="cpu")
    cache_f0_xj = caches["f0_xj"]
    cache_delta_xi = caches["delta_xi"]
    base_lm = caches["base_lm"]  # contains tokenizer

    train_examples = []
    for p in pairs_train:
        ex_i = idx_i.get(p["uid_i"])
        ex_j = idx_j.get(p["uid_j"])
        if ex_i is None or ex_j is None:
            continue
        train_examples.append({"i": ex_i, "j": ex_j, "z": int(p["z"])})

    fc = train_logit(
        train_examples,
        base_lm=base_lm,
        cache_f0_xj=cache_f0_xj,
        cache_delta_xi=cache_delta_xi,
        epochs=cfg["forecasting"]["logit_based"]["epochs"],
        lr=cfg["forecasting"]["logit_based"]["lr"],
        encoder_name=cfg["forecasting"]["encoder"]["name"],
        proj_dim=cfg["forecasting"]["encoder"]["proj_dim"],
    )
    torch.save(fc.state_dict(), os.path.join(args.out_dir, "logit.pt"))
    print("[train] saved logit.pt")


if __name__ == "__main__":
    main()
