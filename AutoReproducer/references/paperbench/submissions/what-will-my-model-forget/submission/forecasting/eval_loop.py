"""Evaluation utilities — F1 / precision / recall, ID/OOD splits, streams.

Per §4.1 (Metrics): "We report F1 scores for binary forgetting prediction."
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Iterable


# ---------------------------------------------------------------------
def precision_recall(
    preds: Iterable[int], labels: Iterable[int]
) -> tuple[float, float]:
    preds = list(preds)
    labels = list(labels)
    tp = sum(int(p == 1 and y == 1) for p, y in zip(preds, labels))
    fp = sum(int(p == 1 and y == 0) for p, y in zip(preds, labels))
    fn = sum(int(p == 0 and y == 1) for p, y in zip(preds, labels))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return prec, rec


def f1_score(preds: Iterable[int], labels: Iterable[int]) -> tuple[float, float, float]:
    """Return (F1, precision, recall) on the binary forgetting label."""
    p, r = precision_recall(preds, labels)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return f1, p, r


# ---------------------------------------------------------------------
def evaluate_forecaster(
    predict_fn: Callable[[str, str, str, str, str], int],
    test_pairs: list[tuple[dict, dict, int]],  # (i, j, z) where i, j carry x, y, uid
) -> dict[str, float]:
    """Compute F1 / precision / recall over `test_pairs`."""
    preds, labels = [], []
    for ex_i, ex_j, z in test_pairs:
        p = predict_fn(ex_i["x"], ex_i["y"], ex_j["x"], ex_j["y"], ex_j["uid"])
        preds.append(int(p))
        labels.append(int(z))
    f1, prec, rec = f1_score(preds, labels)
    return {"f1": f1 * 100.0, "precision": prec * 100.0, "recall": rec * 100.0}


# ---------------------------------------------------------------------
def evaluate_running_metrics(
    stream_predictions: list[tuple[int, int]],
) -> list[tuple[float, float, float]]:
    """Per Figure 3: average F1, P, R up to time-step t.

    `stream_predictions` is a list of (pred, label) in stream order.
    Returns a list of (f1_t, p_t, r_t) for t = 1..N.
    """
    seq = []
    preds: list[int] = []
    labels: list[int] = []
    for p, y in stream_predictions:
        preds.append(int(p))
        labels.append(int(y))
        seq.append(f1_score(preds, labels))
    return seq
