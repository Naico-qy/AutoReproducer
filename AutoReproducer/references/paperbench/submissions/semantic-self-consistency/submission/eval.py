"""
Evaluation entry-point for Semantic Self-Consistency.

Pipeline (paper §4):
  1. For each test example, sample k=10 chain-of-thought rationales from
     the generator (gpt-3.5-turbo or gpt-4o-mini per addendum).
  2. Parse the final answer of each rationale ("the answer is ..." -- addendum).
  3. Embed each rationale with the dataset-specific BERT featurizer
     (SciBERT for AQuA-RAT/SVAMP, RoBERTa for StrategyQA).
  4. For every method in `methods` (top_prob, sc_baseline, cpw, scw,
     isolation_forest, knn_outlier, ocsvm), aggregate to a single answer
     and score against the gold answer.
  5. Write `metrics.json` to /output (paperbench reads this).

Usage:
  python eval.py --config configs/default.yaml --dataset svamp --model gpt-4o-mini
  python eval.py --config configs/default.yaml --smoke   # tiny end-to-end run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

# Allow `python eval.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.loader import load_dataset_split, build_user_prompt  # noqa: E402
from utils.parse import extract_answer, score_prediction  # noqa: E402


# ----------------------------------------------------------------------------- #
def _answer_kind(dataset: str) -> str:
    return {
        "aqua_rat": "letter",
        "svamp": "number",
        "strategyqa": "boolean",
    }[dataset]


def _ensure_output_dir(path: str) -> Path:
    out = Path(path)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError):
        # Fall back to local ./output if /output is not writable
        out = Path("./output")
        out.mkdir(parents=True, exist_ok=True)
    return out


def run_one_dataset(
    cfg: dict,
    dataset: str,
    model_name: str,
    n_examples: int | None = None,
    methods: list[str] | None = None,
) -> dict:
    """Run the full eval loop on one (dataset, generator) pair.

    Returns a metrics dict {method: accuracy}.
    """
    from model import OpenAIGenerator, HFGenerator, Featurizer, SemanticConsistency

    # ---- 1. Build generator ----
    is_openai = model_name.startswith(("gpt-", "openai/"))
    if is_openai:
        gen = OpenAIGenerator(model=model_name)
        sys_prompt = cfg["generators"][0]["api_kwargs"].get("system_prompt")
    else:
        gen = HFGenerator(model_name)
        sys_prompt = None

    # ---- 2. Build featurizer ----
    feat_name = cfg["featurizers"][dataset]
    feat = Featurizer(feat_name)

    # ---- 3. Load test split ----
    ds_cfg = cfg["datasets"][dataset]
    n = n_examples if n_examples is not None else ds_cfg.get("n_examples")
    examples = load_dataset_split(dataset, ds_cfg.get("split", "test"), n_examples=n)
    print(f"[{dataset}] {len(examples)} examples loaded.", flush=True)

    # ---- 4. Generation params ----
    g = cfg["generation"]
    k = g["num_samples"]
    temperature = g["temperature"]
    max_new = g["max_new_tokens"][dataset]

    methods = methods or cfg["methods"]
    correct = {m: 0 for m in methods}
    total = {m: 0 for m in methods}  # only count examples where method
    # produced an answer (addendum line 7)
    per_example_log = []

    kind = _answer_kind(dataset)
    out_cfg = cfg["outlier"]

    for ex in tqdm(examples, desc=f"{model_name}/{dataset}"):
        prompt = build_user_prompt(dataset, ex["question"])
        # ---- generate k rationales ----
        try:
            rationales = gen.generate(
                prompt,
                k=k,
                temperature=temperature,
                max_new_tokens=max_new,
                system_prompt=sys_prompt,
            )
        except Exception as e:
            print(f"[warn] generation failed on {ex['id']}: {e}", flush=True)
            rationales = ["" for _ in range(k)]

        # ---- parse final answers ----
        answers = [extract_answer(r) for r in rationales]

        # ---- embed rationales ----
        emb = (
            feat.encode(rationales)
            if any(r for r in rationales)
            else np.zeros((len(rationales), 1), dtype=np.float32)
        )

        # ---- run each aggregation method ----
        ex_results = {"id": ex["id"], "gold": ex["answer"]}
        for m in methods:
            kwargs = {}
            if m == "isolation_forest":
                kwargs = out_cfg["isolation_forest"]
            elif m == "knn_outlier":
                kwargs = out_cfg["knn"]
            elif m == "ocsvm":
                kwargs = out_cfg["one_class_svm"]
            res = SemanticConsistency.vote(m, emb, answers, **kwargs)
            pred = res.answer

            # Per addendum line 7: top_prob (and ALL methods on AQuA-RAT)
            # only counts examples where an answer could be parsed.
            countable = (
                (m == "top_prob" and pred is not None)
                or (dataset == "aqua_rat" and pred is not None)
                or (m != "top_prob" and dataset != "aqua_rat")
            )
            if countable:
                total[m] += 1
                correct[m] += score_prediction(pred, ex["answer"], kind=kind)
            ex_results[m] = pred
        per_example_log.append(ex_results)

    metrics = {m: round(100.0 * correct[m] / max(1, total[m]), 2) for m in methods}
    metrics["_total"] = total
    metrics["_correct"] = correct
    metrics["_n_examples"] = len(examples)
    metrics["_dataset"] = dataset
    metrics["_model"] = model_name
    return metrics, per_example_log


def main() -> None:
    p = argparse.ArgumentParser(description="Semantic Self-Consistency eval")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--dataset", choices=["aqua_rat", "svamp", "strategyqa", "all"], default="all"
    )
    p.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="Generator model (gpt-3.5-turbo / gpt-4o-mini)",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Which aggregation methods to evaluate",
    )
    p.add_argument(
        "--n-examples", type=int, default=None, help="Truncate dataset for a smoke run"
    )
    p.add_argument(
        "--smoke", action="store_true", help="Tiny end-to-end run (n=2 per dataset)."
    )
    p.add_argument("--output", default=None, help="Override output dir.")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    out_root = _ensure_output_dir(args.output or cfg.get("output_dir", "/output"))

    n = 2 if args.smoke else args.n_examples
    datasets = (
        ["aqua_rat", "svamp", "strategyqa"] if args.dataset == "all" else [args.dataset]
    )

    all_metrics: dict[str, dict] = {}
    for d in datasets:
        try:
            m, log = run_one_dataset(
                cfg, d, args.model, n_examples=n, methods=args.methods
            )
            all_metrics[f"{args.model}/{d}"] = m
            (out_root / f"log_{args.model.replace('/', '_')}_{d}.json").write_text(
                json.dumps(log, indent=2)
            )
            print(json.dumps(m, indent=2), flush=True)
        except Exception as e:
            print(f"[error] {d} failed: {e}", flush=True)
            all_metrics[f"{args.model}/{d}"] = {"error": str(e)}

    metrics_path = out_root / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nWrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
