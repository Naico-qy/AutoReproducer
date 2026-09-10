"""SQuAD-2.0 style Exact-Match scorer (per addendum).

The addendum specifies that predictions are graded with the EM scorer of
the SQuAD 2.0 evaluation script. We re-implement the canonical
normalisation pipeline (lowercase, strip articles, strip punctuation,
collapse whitespace) so the scorer is self-contained and identical to
what the paper used.
"""

from __future__ import annotations

import re
import string
from typing import Iterable


# ---------------------------------------------------------------------
def _normalize_answer(s: str) -> str:
    """SQuAD 2.0 normalisation."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def squad_em(prediction: str, ground_truth: str) -> int:
    """Return 1 if prediction exactly matches `ground_truth` after normalisation."""
    if prediction is None:
        return 0
    return int(_normalize_answer(prediction) == _normalize_answer(ground_truth))


# alias used by the paper's notation EM_{D, f}
def exact_match_score(prediction: str, ground_truth: str) -> int:
    return squad_em(prediction, ground_truth)


# ---------------------------------------------------------------------
def dataset_em(predictions: Iterable[str], targets: Iterable[str]) -> float:
    """Per §2: EM_{D,f} := |{(x,y) in D | f(x) = y}| / |D|."""
    preds = list(predictions)
    tgts = list(targets)
    if not tgts:
        return 0.0
    return sum(squad_em(p, t) for p, t in zip(preds, tgts)) / len(tgts)


def em_drop_ratio(em_after: float, em_before: float) -> float:
    """Per §2: (EM_{D_PT, f_i} - EM_{D_PT, f_0}) / EM_{D_PT, f_0}.

    Negative values indicate forgetting (EM after refinement is lower).
    """
    if em_before <= 0:
        return 0.0
    return (em_after - em_before) / em_before


def edit_success_rate(predictions: Iterable[str], targets: Iterable[str]) -> float:
    """Per §2: |{(x_i,y_i) in D_R | f_i(x_i) = y_i}| / |D_R|."""
    return dataset_em(predictions, targets)
