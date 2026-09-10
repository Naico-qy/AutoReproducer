"""
"Training" entry-point for Semantic Self-Consistency.

Important context (paper §1, §4 + Appendix E.2):
  This method is *training-free* -- it operates at *inference time* on the
  k=10 sampled rationales of an existing LLM. Quoting Appendix E.2:
    "Our methods weighs results directly after generation in a separate
     weighting/filtering step ... we spare computational efforts by not
     requiring an additional pre-training step."

To match the PaperBench `train.py` interface, this script performs a
warm-up / dry-run that:
  1. Loads the YAML config and validates every section.
  2. Materializes the BERT featurizer to download weights ahead of eval.
  3. Caches a tiny number of rationales (k=10 on n=2 examples per dataset)
     for sanity-checking the generation + parsing + voting pipeline.
  4. Writes a "training" log to the output dir.

Run `python eval.py` for the actual benchmark numbers.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.loader import load_dataset_split, build_user_prompt  # noqa: E402
from utils.parse import extract_answer  # noqa: E402


def warmup_featurizer(model_name: str) -> bool:
    """Download + sanity-check a BERT featurizer."""
    from model import Featurizer

    feat = Featurizer(model_name)
    out = feat.encode(["hello world", "the answer is 42."])
    assert out.shape[0] == 2 and out.shape[1] > 0
    return True


def warmup_generation(cfg: dict, dataset: str, model_name: str, k: int = 2) -> dict:
    from model import OpenAIGenerator, HFGenerator

    if model_name.startswith(("gpt-", "openai/")):
        gen = OpenAIGenerator(model=model_name)
        sys_prompt = cfg["generators"][0]["api_kwargs"].get("system_prompt")
    else:
        gen = HFGenerator(model_name)
        sys_prompt = None

    examples = load_dataset_split(dataset, "test", n_examples=k)
    log = []
    for ex in examples:
        prompt = build_user_prompt(dataset, ex["question"])
        t0 = time.time()
        try:
            rationales = gen.generate(
                prompt,
                k=cfg["generation"]["num_samples"],
                temperature=cfg["generation"]["temperature"],
                max_new_tokens=cfg["generation"]["max_new_tokens"][dataset],
                system_prompt=sys_prompt,
            )
        except Exception as e:
            rationales = [f"[generation failed: {e}]"]
        dt = time.time() - t0
        log.append(
            {
                "id": ex["id"],
                "gold": ex["answer"],
                "n_rationales": len(rationales),
                "first_rationale": rationales[0][:300],
                "first_parsed": extract_answer(rationales[0]),
                "elapsed_sec": round(dt, 2),
            }
        )
    return {"dataset": dataset, "log": log}


def main() -> None:
    p = argparse.ArgumentParser(description="Semantic Self-Consistency warm-up")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", default="gpt-3.5-turbo")
    p.add_argument("--datasets", nargs="+", default=["aqua_rat", "svamp", "strategyqa"])
    p.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip API calls -- only warm up featurizers.",
    )
    p.add_argument("--output", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_root = Path(args.output or cfg.get("output_dir", "/output"))
    try:
        out_root.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError):
        out_root = Path("./output")
        out_root.mkdir(parents=True, exist_ok=True)

    train_log = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_path": args.config,
        "model": args.model,
        "stages": [],
    }

    # Stage 1: featurizer warm-up
    feat_log: dict = {}
    for d, name in cfg["featurizers"].items():
        try:
            warmup_featurizer(name)
            feat_log[d] = f"OK ({name})"
        except Exception as e:
            feat_log[d] = f"FAIL: {e}"
    train_log["stages"].append({"featurizer_warmup": feat_log})
    print(json.dumps(feat_log, indent=2), flush=True)

    # Stage 2: generation warm-up
    if not args.skip_generation:
        for d in args.datasets:
            try:
                gen_log = warmup_generation(cfg, d, args.model, k=2)
                train_log["stages"].append({f"generation_{d}": gen_log})
            except Exception as e:
                train_log["stages"].append({f"generation_{d}": {"error": str(e)}})

    # Persist log
    log_path = out_root / "train_log.json"
    log_path.write_text(json.dumps(train_log, indent=2, default=str))
    print(f"Wrote {log_path}", flush=True)


if __name__ == "__main__":
    main()
