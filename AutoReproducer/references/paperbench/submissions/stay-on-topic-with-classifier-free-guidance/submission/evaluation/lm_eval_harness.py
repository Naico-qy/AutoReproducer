"""Zero-shot LM-Evaluation-Harness style scoring -- §3.1.

The paper inherits the harness defaults (addendum):
    * Multiple-choice tasks are scored by `acc_norm` -- the option with the
      highest length-normalised loglikelihood is chosen.
    * Open-ended tasks (TriviaQA, LAMBADA) are scored by exact-match on
      the greedy continuation.

We compute these metrics with **CFG-enhanced** loglikelihoods (Eq. 7),
which is precisely the technique the paper evaluates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
from tqdm import tqdm

from data.loader import BenchmarkSample, load_zero_shot
from model.architecture import CFGSampler


# ---------------------------------------------------------------------------
# Multiple-choice scoring
# ---------------------------------------------------------------------------
def score_choices_with_cfg(
    sampler: CFGSampler,
    context: str,
    choices: List[str],
    length_normalize: bool = True,
) -> List[float]:
    """Return one score per candidate continuation.

    Implements `acc_norm`-style scoring: log P(choice | context) divided by
    the number of bytes (LM-Eval convention).
    """
    scores = []
    for c in choices:
        logp = sampler.loglikelihood(context, " " + c.lstrip())
        if length_normalize:
            n_bytes = max(len(c), 1)
            logp = logp / n_bytes
        scores.append(float(logp))
    return scores


# ---------------------------------------------------------------------------
# Top-level evaluator
# ---------------------------------------------------------------------------
@dataclass
class ZeroShotResult:
    dataset: str
    gamma: float
    metric: str
    score: float
    n: int


def _exact_match(pred: str, gold: str) -> bool:
    return pred.strip().lower() == gold.strip().lower()


@torch.no_grad()
def evaluate_zero_shot(
    sampler: CFGSampler,
    name: str,
    hf_path: str,
    config: Optional[str],
    metric: str,
    gamma: float,
    max_samples: Optional[int] = None,
    split: str = "validation",
) -> ZeroShotResult:
    """Run zero-shot evaluation on a single benchmark."""
    sampler.guidance_scale = gamma
    correct = 0
    total = 0
    samples: Iterable[BenchmarkSample] = load_zero_shot(
        name, hf_path, config, split=split
    )

    for i, sample in enumerate(tqdm(samples, desc=f"{name} γ={gamma}")):
        if max_samples is not None and i >= max_samples:
            break
        if sample.choices:  # multiple-choice
            scores = score_choices_with_cfg(
                sampler,
                sample.context,
                sample.choices,
                length_normalize=metric == "acc_norm",
            )
            pred_idx = int(max(range(len(scores)), key=lambda j: scores[j]))
            correct += int(pred_idx == sample.answer_idx)
        else:  # open-ended
            pred = sampler.generate(
                sample.context,
                max_new_tokens=32,
                do_sample=False,
            )
            correct += int(_exact_match(pred, sample.answer_text or ""))
        total += 1

    return ZeroShotResult(
        dataset=name,
        gamma=gamma,
        metric=metric,
        score=correct / max(total, 1),
        n=total,
    )
