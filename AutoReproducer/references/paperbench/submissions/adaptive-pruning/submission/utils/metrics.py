"""Metric helpers used across train.py / eval.py.

Implements:
  * SQuAD-v2 F1 (paper §5.1)
  * ROUGE-1/2/L for CNN/DM (paper §5.1)
  * Time-to-Accuracy (TTA) following Coleman et al., 2019 — referenced
    in §5.3 of the paper.
"""

from __future__ import annotations

import string
import re
from collections import Counter
from typing import Dict, List, Tuple


# --------------------------------------------------------------------------- #
def _normalize_answer(s: str) -> str:
    """Lower-case, strip punctuation/articles/whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in set(string.punctuation))
    return " ".join(s.split())


def squad_f1(pred: str, gold: str) -> float:
    p_toks = _normalize_answer(pred).split()
    g_toks = _normalize_answer(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_toks)
    recall = num_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def squad_em(pred: str, gold: str) -> float:
    return float(_normalize_answer(pred) == _normalize_answer(gold))


# --------------------------------------------------------------------------- #
def rouge_n(pred: str, ref: str, n: int = 1) -> float:
    p_toks = pred.split()
    r_toks = ref.split()
    p_ngrams = [tuple(p_toks[i : i + n]) for i in range(len(p_toks) - n + 1)]
    r_ngrams = [tuple(r_toks[i : i + n]) for i in range(len(r_toks) - n + 1)]
    if not p_ngrams or not r_ngrams:
        return 0.0
    overlap = sum((Counter(p_ngrams) & Counter(r_ngrams)).values())
    return overlap / len(r_ngrams)


def _lcs(a: List[str], b: List[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(pred: str, ref: str) -> float:
    p_toks = pred.split()
    r_toks = ref.split()
    if not p_toks or not r_toks:
        return 0.0
    lcs = _lcs(p_toks, r_toks)
    p = lcs / len(p_toks)
    r = lcs / len(r_toks)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# --------------------------------------------------------------------------- #
def time_to_accuracy(history: List[Dict], target_acc: float) -> float:
    """Return the wall-time (seconds) of the first eval where accuracy ≥ target.

    Coleman et al., 2019; cited in §5.3 of APT.  Returns -1 if never reached.
    """
    for ev in history:
        if ev.get("accuracy", 0.0) >= target_acc:
            return float(ev.get("wall_time_sec", -1))
    return -1.0
