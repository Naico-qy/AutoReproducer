"""
Online-adaptation training entrypoint for BBox-Adapter.

Implements Algorithm 1 of Sun et al. (ICML 2024):

    for t = 0..T-1:
        for i = 1..N:
            sample candidates {y_hat_{i,m}} ~ p_theta_t(y | x_i)         (Eq. 4)
            update positive y_i+^{(t)} via SEL                            (Eq. 5)
            update negatives y_i-^{(t)} = candidates \ {y_+}              (Eq. 6)
        compute ranking-NCE gradient (Eq. 3) and step AdamW               (Eq. 7)

Defaults from `configs/default.yaml` mirror Appendix H.2 of the paper:
DeBERTa-v3-base / large adapter, lr 5e-6, batch 64, 6000 steps, AdamW,
weight-decay 0.01, beam size 3, temperature 1.0.

Run example:

    python train.py --config configs/default.yaml --task gsm8k --feedback combined
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import yaml

from data import (
    Example,
    FeedbackMode,
    PositivePool,
    NegativePool,
    answers_match,
    extract_final_answer,
    load_prompt,
    load_task,
)
from data.pools import select_positive
from model import BBoxAdapter, build_llm_client, sentence_beam_search
from model.architecture import AdapterConfig
from model.llm_client import LLMConfig
from model.loss import compute_nce_batch_loss


# ----------------------------------------------------------------------
# CLI / config
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BBox-Adapter online training")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--task", type=str, default=None, help="overrides cfg.data.task")
    p.add_argument(
        "--feedback",
        type=str,
        default=None,
        choices=["ground_truth", "ai_feedback", "combined"],
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="override cfg.train.num_steps (smoke runs)",
    )
    p.add_argument(
        "--iterations", type=int, default=None, help="override cfg.train.num_iterations"
    )
    p.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="cap the number of training questions",
    )
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument(
        "--llm-backend", type=str, default=None, choices=["openai", "hf", "dummy"]
    )
    p.add_argument(
        "--smoke", action="store_true", help="enable a tiny end-to-end smoke run"
    )
    return p.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def get_device():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    except ImportError:
        return None


def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ----------------------------------------------------------------------
# Algorithm 1 — online adaptation
# ----------------------------------------------------------------------
def online_adaptation_loop(cfg: dict, args: argparse.Namespace) -> Dict[str, float]:
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    set_seed(cfg.get("seed", 42))
    device = get_device()

    # ------------------------------------------------------------------
    # Resolve overrides
    # ------------------------------------------------------------------
    task = args.task or cfg["data"]["task"]
    feedback = args.feedback or cfg["online"]["feedback"]
    num_steps = args.steps or cfg["train"]["num_steps"]
    num_iters = args.iterations or cfg["train"]["num_iterations"]
    max_train = args.max_train or (
        cfg["data"]["num_train_examples"]
        if cfg["data"]["num_train_examples"] > 0
        else -1
    )
    if args.smoke:
        num_iters = max(1, min(num_iters, 1))
        num_steps = max(8, min(num_steps, 32))
        max_train = max(8, min(max_train if max_train > 0 else 16, 16))

    output_dir = args.output_dir or cfg["output"]["fallback_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Build adapter
    # ------------------------------------------------------------------
    adapter_cfg = AdapterConfig(**cfg["adapter"])
    adapter = BBoxAdapter(adapter_cfg).to(device)
    print(
        f"[info] adapter backbone={adapter_cfg.backbone}  "
        f"params={adapter.num_parameters / 1e6:.1f}M  device={device}"
    )

    # ------------------------------------------------------------------
    # Build LLM client
    # ------------------------------------------------------------------
    llm_backend = args.llm_backend or cfg["llm"]["backend"]
    llm_cfg = LLMConfig(
        backend=llm_backend,
        name=cfg["llm"]["name"],
        temperature=cfg["llm"]["temperature"],
        max_tokens=cfg["llm"]["max_tokens"],
    )
    llm = build_llm_client(llm_cfg)
    print(f"[info] llm backend={llm.name}")

    # ------------------------------------------------------------------
    # Load data + prompt
    # ------------------------------------------------------------------
    examples: List[Example] = load_task(
        task,
        split="train",
        cache_dir=cfg["data"]["data_dir"],
        max_examples=max_train,
        seed=cfg.get("seed", 42),
    )
    prompt_template = _try_load_prompt(task, cfg["data"]["prompt_dir"])
    ai_prompt = _try_load_ai_prompt(task, cfg["data"]["prompt_dir"])
    print(f"[info] loaded {len(examples)} {task} examples")

    # ------------------------------------------------------------------
    # Pools & optimiser
    # ------------------------------------------------------------------
    positives = PositivePool()
    negatives = NegativePool(
        capacity=cfg["online"].get("candidates_per_question", 3) * 4
    )

    # Initialise pools (Algorithm 1, "Initialization" paragraph).
    _initialise_pools(
        examples=examples,
        adapter=adapter,
        llm=llm,
        prompt_template=prompt_template,
        ai_prompt=ai_prompt,
        feedback=feedback,
        positives=positives,
        negatives=negatives,
        num_candidates=cfg["online"]["candidates_per_question"],
        device=device,
    )

    optim = AdamW(
        adapter.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
        eps=cfg["optim"]["eps"],
    )
    warmup = cfg["optim"]["warmup_steps"]

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (num_steps - step) / max(1, num_steps - warmup))

    sched = LambdaLR(optim, lr_lambda)

    # ------------------------------------------------------------------
    # Outer loop — Algorithm 1.
    # ------------------------------------------------------------------
    metrics: Dict[str, float] = {}
    global_step = 0
    rng = random.Random(cfg.get("seed", 42))

    for t in range(num_iters):
        print(f"\n=== Iteration {t + 1}/{num_iters} ===")

        # ---- (1) Sample candidates from p_theta_t and refresh pools.
        if t > 0 and cfg["online"].get("refresh_candidates_each_iter", True):
            _refresh_pools(
                examples=examples,
                adapter=adapter,
                llm=llm,
                prompt_template=prompt_template,
                ai_prompt=ai_prompt,
                feedback=feedback,
                positives=positives,
                negatives=negatives,
                cfg=cfg,
                device=device,
            )

        # ---- (3) Update adapter parameters via NCE (Eq. 3 / Eq. 7).
        steps_per_iter = max(1, num_steps // num_iters)
        adapter.train()
        for inner in range(steps_per_iter):
            batch_qs, batch_pos, batch_neg = _sample_training_batch(
                examples=examples,
                positives=positives,
                negatives=negatives,
                batch_size=cfg["train"]["batch_size"],
                neg_per_question=cfg["online"]["candidates_per_question"],
                rng=rng,
            )
            if not batch_qs:
                continue

            optim.zero_grad()
            stats = compute_nce_batch_loss(
                adapter=adapter,
                questions=batch_qs,
                positives=batch_pos,
                negatives=batch_neg,
                alpha=cfg["train"]["alpha"],
                device=device,
            )
            stats["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                adapter.parameters(), cfg["optim"]["max_grad_norm"]
            )
            optim.step()
            sched.step()
            global_step += 1

            if global_step % cfg["train"]["log_every"] == 0:
                print(
                    f"step {global_step:5d}"
                    f"  loss={stats['loss'].item():.4f}"
                    f"  nce={stats['nce'].item():.4f}"
                    f"  reg={stats['reg'].item():.4f}"
                    f"  acc@1={stats['acc@1'].item():.3f}"
                    f"  lr={sched.get_last_lr()[0]:.2e}"
                )

            metrics = {
                "loss": stats["loss"].item(),
                "nce": stats["nce"].item(),
                "reg": stats["reg"].item(),
                "acc@1": stats["acc@1"].item(),
                "step": global_step,
                "iteration": t + 1,
                "num_train_examples": len(examples),
                "task": task,
                "feedback": feedback,
            }

    # ------------------------------------------------------------------
    # Save adapter + metrics
    # ------------------------------------------------------------------
    ckpt_path = os.path.join(output_dir, cfg["output"]["ckpt_name"])
    adapter.save(ckpt_path)
    print(f"[info] saved adapter -> {ckpt_path}")

    metrics_path = os.path.join(output_dir, cfg["output"]["metrics_name"])
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[info] wrote metrics -> {metrics_path}")
    return metrics


# ----------------------------------------------------------------------
# Pool helpers
# ----------------------------------------------------------------------
def _initialise_pools(
    examples,
    adapter,
    llm,
    prompt_template: str,
    ai_prompt: str,
    feedback: str,
    positives: PositivePool,
    negatives: NegativePool,
    num_candidates: int,
    device,
):
    """Algorithm 1, Initialization paragraph.

    The positive set comes from ground-truth (or AI feedback / combined)
    and the negative set is drawn from p_theta_0 (random adapter)."""
    print(f"[info] initialising pools with feedback={feedback}")
    for ex in examples:
        prompt = prompt_template.replace("<QUESTION>", ex.question)
        cands = llm.generate_complete(prompt, n=num_candidates)
        # Synthetic ground-truth rationale acts as the seed positive.
        seed_pos = ex.rationale or ex.answer
        new_pos, negs = select_positive(
            mode=feedback,
            question=ex.question,
            gold_answer=ex.answer,
            prev_positive=seed_pos,
            candidates=cands,
            llm_client=llm,
            ai_feedback_prompt=ai_prompt,
        )
        positives.set(ex.qid, new_pos)
        negatives.add(ex.qid, negs, pos=new_pos)


def _refresh_pools(
    examples,
    adapter,
    llm,
    prompt_template: str,
    ai_prompt: str,
    feedback: str,
    positives: PositivePool,
    negatives: NegativePool,
    cfg: dict,
    device,
):
    print("[info] refreshing pools via adapted inference (Eq. 4)")
    for ex in examples:
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
        # Treat the beam-search result as the new candidate; mix with
        # additional shallow samples to populate the negative pool.
        prompt = prompt_template.replace("<QUESTION>", ex.question)
        extra = llm.generate_complete(
            prompt, n=cfg["online"]["candidates_per_question"]
        )
        cands = [gen] + extra
        prev = positives.get(ex.qid)
        new_pos, negs = select_positive(
            mode=feedback,
            question=ex.question,
            gold_answer=ex.answer,
            prev_positive=prev,
            candidates=cands,
            llm_client=llm,
            ai_feedback_prompt=ai_prompt,
        )
        positives.set(ex.qid, new_pos)
        negatives.add(ex.qid, negs, pos=new_pos)


def _sample_training_batch(
    examples,
    positives: PositivePool,
    negatives: NegativePool,
    batch_size: int,
    neg_per_question: int,
    rng: random.Random,
):
    pool_qids = [ex.qid for ex in examples if positives.get(ex.qid) is not None]
    if not pool_qids:
        return [], [], []
    n = min(batch_size, len(pool_qids))
    chosen = rng.sample(pool_qids, n)
    qid_to_ex = {ex.qid: ex for ex in examples}
    qs, pos, neg = [], [], []
    for qid in chosen:
        ex = qid_to_ex[qid]
        p = positives.get(qid)
        ns = negatives.sample(qid, neg_per_question, rng=rng)
        if not ns:
            ns = ["No answer."] * neg_per_question
        qs.append(ex.question)
        pos.append(p)
        neg.append(ns)
    return qs, pos, neg


# ----------------------------------------------------------------------
# Prompt loaders with graceful fallback
# ----------------------------------------------------------------------
def _try_load_prompt(task: str, prompt_dir: str) -> str:
    try:
        return load_prompt(task, prompt_dir)
    except Exception:
        return "Q: <QUESTION>\nA: Let's think step by step."


def _try_load_ai_prompt(task: str, prompt_dir: str) -> str:
    fname = {
        "strategyqa": "ai_feedback_strategyqa.txt",
        "gsm8k": "ai_feedback_gsm8k.txt",
        "truthfulqa": "ai_feedback_truthfulqa.txt",
        "scienceqa": "ai_feedback_scienceqa.txt",
    }.get(task, "ai_feedback_strategyqa.txt")
    path = os.path.join(prompt_dir, fname)
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return ""


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    metrics = online_adaptation_loop(cfg, args)
    print("\n[info] training done.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
