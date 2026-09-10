"""
Evaluation entrypoint for BBox-Adapter.

Loads a trained adapter checkpoint and runs `sentence_beam_search`
(§3.3 / Eq. 4) against the task's test split, then computes:

  * Accuracy  — exact-match on the final answer extracted via `####`.
  * For TruthfulQA: True+Info (lower-bounded by exact match here; the
    paper uses the official judge models, which are dataset-specific).

Run example:

    python eval.py --config configs/default.yaml \
                   --task gsm8k --ckpt outputs/adapter.pt
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import yaml

from data import (
    Example,
    answers_match,
    extract_final_answer,
    load_prompt,
    load_task,
)
from model import BBoxAdapter, build_llm_client
from model.architecture import AdapterConfig
from model.inference import sentence_beam_search, single_step_inference
from model.llm_client import LLMConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--task", type=str, default=None)
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to a saved adapter (defaults to <output_dir>/adapter.pt)",
    )
    p.add_argument(
        "--max-eval", type=int, default=None, help="cap on the number of test questions"
    )
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument(
        "--llm-backend", type=str, default=None, choices=["openai", "hf", "dummy"]
    )
    p.add_argument(
        "--single-step", action="store_true", help="use single-step inference (Table 4)"
    )
    p.add_argument(
        "--smoke", action="store_true", help="evaluate on only a handful of questions"
    )
    return p.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except ImportError:
        return None


def evaluate(cfg: dict, args: argparse.Namespace) -> Dict[str, float]:
    import torch

    device = get_device()
    task = args.task or cfg["data"]["task"]
    output_dir = args.output_dir or cfg["output"]["fallback_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # ----- adapter
    adapter_cfg = AdapterConfig(**cfg["adapter"])
    adapter = BBoxAdapter(adapter_cfg).to(device)
    ckpt = args.ckpt or os.path.join(output_dir, cfg["output"]["ckpt_name"])
    if os.path.exists(ckpt):
        try:
            loaded = BBoxAdapter.load(ckpt, map_location=str(device))
            adapter.load_state_dict(loaded.state_dict())
            print(f"[info] loaded adapter weights from {ckpt}")
        except Exception as e:
            print(
                f"[warn] could not load checkpoint ({e}); evaluating "
                f"a freshly-initialised adapter"
            )
    adapter.eval()

    # ----- LLM
    llm_backend = args.llm_backend or cfg["llm"]["backend"]
    llm_cfg = LLMConfig(
        backend=llm_backend,
        name=cfg["llm"]["name"],
        temperature=cfg["llm"]["temperature"],
        max_tokens=cfg["llm"]["max_tokens"],
    )
    llm = build_llm_client(llm_cfg)

    # ----- data
    max_eval = args.max_eval or (
        cfg["data"]["num_eval_examples"] if cfg["data"]["num_eval_examples"] > 0 else -1
    )
    if args.smoke:
        max_eval = max(4, min(max_eval if max_eval > 0 else 8, 8))

    examples: List[Example] = load_task(
        task,
        split="test",
        cache_dir=cfg["data"]["data_dir"],
        max_examples=max_eval,
        seed=cfg.get("seed", 42),
    )
    prompt_template = _try_load_prompt(task, cfg["data"]["prompt_dir"])
    print(
        f"[info] evaluating on {len(examples)} {task} examples (single_step={args.single_step})"
    )

    # ----- evaluation loop
    n_correct = 0
    predictions = []
    for i, ex in enumerate(examples):
        if args.single_step or not cfg["inference"].get("full_step", True):
            gen = single_step_inference(
                adapter,
                llm,
                ex.question,
                prompt_template,
                num_candidates=cfg["inference"]["num_candidates_per_step"],
                device=device,
            )
        else:
            gen = sentence_beam_search(
                adapter=adapter,
                llm_client=llm,
                question=ex.question,
                prompt_template=prompt_template,
                beam_size=cfg["inference"]["beam_size"],
                num_candidates_per_step=cfg["inference"]["num_candidates_per_step"],
                max_steps=cfg["inference"]["max_steps"],
                stop_token=cfg["inference"]["stop_token"],
                device=device,
            )
        pred_ans = extract_final_answer(gen)
        ok = answers_match(pred_ans, ex.answer)
        if ok:
            n_correct += 1
        predictions.append(
            {
                "qid": ex.qid,
                "question": ex.question,
                "gold_answer": ex.answer,
                "prediction": gen,
                "predicted_answer": pred_ans,
                "correct": ok,
            }
        )
        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{len(examples)}  acc={n_correct / (i + 1):.3f}")

    acc = n_correct / max(1, len(examples))
    metrics = {
        "task": task,
        "num_eval": len(examples),
        "accuracy": acc,
        "true_plus_info": acc,  # placeholder for TruthfulQA judge
    }

    metrics_path = os.path.join(output_dir, cfg["output"]["metrics_name"])
    # Merge with any pre-existing training metrics so the reproducer
    # can read both.
    existing: Dict[str, object] = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing.update({"eval": metrics})
    existing.update(metrics)  # also expose flat keys
    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"[info] wrote eval metrics -> {metrics_path}")

    preds_path = os.path.join(output_dir, "predictions.jsonl")
    with open(preds_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    print(f"[info] wrote predictions -> {preds_path}")
    return metrics


def _try_load_prompt(task: str, prompt_dir: str) -> str:
    try:
        return load_prompt(task, prompt_dir)
    except Exception:
        return "Q: <QUESTION>\nA: Let's think step by step."


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    metrics = evaluate(cfg, args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
