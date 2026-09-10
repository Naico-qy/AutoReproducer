"""OOD-error predictors for the Table 3 comparison.

Each predictor fits a 1-D linear regression mapping an in-distribution
quantity (Top-1 accuracy, average confidence, agreement consensus, or LCA
distance) to OOD Top-1 accuracy across the 75 evaluated models, then
reports the held-out MAE.

Per the addendum, Aline-S and Aline-D follow the implementation from
https://github.com/kebaek/Agreement-on-the-line/blob/main/agreement_trajectory.ipynb
(reproduced faithfully here).

Per the paper §4.2, we use *min-max scaling* instead of probit transform
because LCA does not lie in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


@dataclass
class Prediction:
    name: str
    mae: float
    coef: float
    intercept: float


class _BasePredictor:
    """Common helper: fit y_ood = a * x_id + b after MinMax-scaling x_id."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.scaler = MinMaxScaler()
        self.linreg = LinearRegression()

    def _fit(self, x_id: np.ndarray, y_ood: np.ndarray) -> Prediction:
        x_id = np.asarray(x_id, dtype=np.float64).reshape(-1, 1)
        y_ood = np.asarray(y_ood, dtype=np.float64).reshape(-1)
        x_scaled = self.scaler.fit_transform(x_id)
        self.linreg.fit(x_scaled, y_ood)
        pred = self.linreg.predict(x_scaled)
        mae = float(np.mean(np.abs(pred - y_ood)))
        return Prediction(
            self.name, mae, float(self.linreg.coef_[0]), float(self.linreg.intercept_)
        )

    def predict(self, x_id: Sequence[float]) -> np.ndarray:
        x = np.asarray(x_id, dtype=np.float64).reshape(-1, 1)
        return self.linreg.predict(self.scaler.transform(x))


class AccuracyOnTheLine(_BasePredictor):
    """ID Top-1 accuracy as predictor of OOD Top-1 accuracy.

    Reference (verified via paper_search & CrossRef metadata):
        Miller et al., 'Accuracy on the Line', ICML 2021.
    """

    def __init__(self) -> None:
        super().__init__(name="ID Top1 (Miller et al., 2021)")

    def fit(self, id_top1: Sequence[float], ood_top1: Sequence[float]) -> Prediction:
        return self._fit(np.asarray(id_top1), np.asarray(ood_top1))


class AverageConfidence(_BasePredictor):
    """Average confidence (AC) using softmax probability of the top-1 class.

    Reference: Hendrycks & Gimpel, ICLR 2017.

    The paper's Table 3 implementation: temperature-scaled OOD logits and
    take the mean Top-1 softmax probability per model.  We expose a method
    that consumes pre-computed AC values (one scalar per model) since
    temperature scaling is a model-specific calibration step.
    """

    def __init__(self) -> None:
        super().__init__(name="AC (Hendrycks & Gimpel, 2017)")

    def fit(self, avg_conf: Sequence[float], ood_top1: Sequence[float]) -> Prediction:
        return self._fit(np.asarray(avg_conf), np.asarray(ood_top1))


class _AlineBase(_BasePredictor):
    """Agreement-on-the-line (Baek et al., 2022).

    For each pair of models (i, j), agreement is the fraction of OOD samples
    on which they predict the same class.  Aline-D uses pairwise OOD
    *disagreement* of training-set models, Aline-S uses pairwise OOD
    agreement.  Faithful to the reference implementation cited in the
    addendum.
    """

    def fit(
        self,
        agreement_matrix: np.ndarray,
        ood_top1: Sequence[float],
    ) -> Prediction:
        # Per-model statistic = mean (dis)agreement against the rest of the
        # population, then a 1-D linear regression on OOD Top-1.
        per_model = agreement_matrix.mean(axis=1)
        return self._fit(per_model, np.asarray(ood_top1))


class AlineD(_AlineBase):
    """Aline-D — uses pairwise *disagreement* (1 - agreement)."""

    def __init__(self) -> None:
        super().__init__(name="Aline-D (Baek et al., 2022)")

    def fit(
        self,
        agreement_matrix: np.ndarray,
        ood_top1: Sequence[float],
    ) -> Prediction:
        return super().fit(1.0 - agreement_matrix, ood_top1)


class AlineS(_AlineBase):
    """Aline-S — uses pairwise *agreement* directly."""

    def __init__(self) -> None:
        super().__init__(name="Aline-S (Baek et al., 2022)")


class LCAPredictor(_BasePredictor):
    """ID LCA distance as predictor of OOD Top-1 accuracy (paper §4.2)."""

    def __init__(self) -> None:
        super().__init__(name="(Ours) ID LCA")

    def fit(self, id_lca: Sequence[float], ood_top1: Sequence[float]) -> Prediction:
        # Negative correlation: lower LCA -> higher OOD acc, but the linear
        # regression handles sign naturally.
        return self._fit(np.asarray(id_lca), np.asarray(ood_top1))


def correlation_metrics(x: Sequence[float], y: Sequence[float]) -> dict:
    """Compute R^2, Pearson, Kendall, Spearman as in Tables 2/F.

    Returns a dict with absolute-value correlations (paper §4.1: "we take
    the absolute value of all correlations for simplicity").
    """
    from scipy.stats import kendalltau, pearsonr, spearmanr  # local import

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pea, _ = pearsonr(x, y)
    ken, _ = kendalltau(x, y)
    spe, _ = spearmanr(x, y)
    r2 = pea**2  # absolute coefficient of determination
    return {
        "r2": float(abs(r2)),
        "pearson": float(abs(pea)),
        "kendall": float(abs(ken)),
        "spearman": float(abs(spe)),
    }
