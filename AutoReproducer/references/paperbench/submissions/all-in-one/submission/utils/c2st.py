"""Classifier Two-Sample Test (C2ST).

Per addendum.md: implemented with a Random Forest of 100 trees. Returns
mean classification accuracy across a 5-fold cross-validation; 0.5 means
the two samples are indistinguishable, 1.0 means perfectly separable.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold


def c2st_random_forest(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
    n_estimators: int = 100,
    n_folds: int = 5,
    seed: int = 0,
) -> float:
    """C2ST accuracy between samples from p and from q."""
    samples_p = np.asarray(samples_p)
    samples_q = np.asarray(samples_q)
    n = min(len(samples_p), len(samples_q))
    samples_p = samples_p[:n]
    samples_q = samples_q[:n]
    X = np.concatenate([samples_p, samples_q], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)], axis=0)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs = []
    for train_idx, test_idx in kf.split(X):
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
        clf.fit(X[train_idx], y[train_idx])
        accs.append(clf.score(X[test_idx], y[test_idx]))
    return float(np.mean(accs))
