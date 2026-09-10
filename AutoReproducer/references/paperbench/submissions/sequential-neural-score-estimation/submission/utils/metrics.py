"""Classifier two-sample test (C2ST) for posterior evaluation.

The paper (Section 5.2) reports the classification-based two-sample test
(C2ST) score (Lopez-Paz & Oquab, 2017) as the primary metric. Per the
addendum, the ``sbibm`` library should be used to implement C2ST with its
default hyperparameters; we therefore wrap that whenever it is available.
We additionally provide a self-contained scikit-learn-based fallback so
the pipeline still runs in environments without sbibm.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def c2st_via_sbibm(
    posterior_samples: torch.Tensor,
    reference_samples: torch.Tensor,
) -> float:
    """C2ST as implemented in sbibm (Lueckmann et al., 2021).

    Returns a value in [0.5, 1] (lower is better; 0.5 = perfect match).
    """
    try:
        from sbibm.metrics import c2st as _c2st_sbibm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sbibm not installed — fall back to `c2st` (sklearn-based)."
        ) from exc
    return float(_c2st_sbibm(posterior_samples, reference_samples).item())


def c2st(
    posterior_samples: torch.Tensor,
    reference_samples: torch.Tensor,
    n_folds: int = 5,
    seed: int = 1,
) -> float:
    """sklearn-based C2ST (fallback when sbibm is unavailable).

    Trains a 2-hidden-layer MLP classifier (matching sbibm's default
    architecture) to discriminate between `posterior_samples` and
    `reference_samples`, and reports the mean cross-validated test accuracy.
    """
    try:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.neural_network import MLPClassifier
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scikit-learn required for c2st fallback.") from exc

    a = posterior_samples.detach().cpu().numpy()
    b = reference_samples.detach().cpu().numpy()
    n = min(a.shape[0], b.shape[0])
    a = a[:n]
    b = b[:n]

    X = np.concatenate([a, b], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(n), np.ones(n)], axis=0).astype(np.int64)

    # Standardise features (sbibm default).
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - mu) / sd

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = []
    for train, test in skf.split(X, y):
        clf = MLPClassifier(
            hidden_layer_sizes=(10 * X.shape[1], 10 * X.shape[1]),
            activation="relu",
            max_iter=10000,
            solver="adam",
            random_state=seed,
        )
        clf.fit(X[train], y[train])
        scores.append(clf.score(X[test], y[test]))
    return float(np.mean(scores))
