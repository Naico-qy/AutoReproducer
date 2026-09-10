"""§3.1 Frequency-Threshold based Forecasting (Eqn. 1).

Per addendum: D_PT here means the *correctly-predicted* subset \\hat{D}_PT.

    g((x_i, y_i), (x_j, y_j)) = 1 [ |{i : z_{ij} = 1}| >= γ ]

γ is tuned to maximize F1 on D_R^Train.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


class ThresholdForecaster:
    """Predict 'forgotten' if x_j has been forgotten ≥ γ times in D_R^Train."""

    def __init__(self, gamma: int = 1):
        self.gamma = gamma
        self.counts: dict[str, int] = defaultdict(int)

    # ----- training -------------------------------------------------
    def fit(self, train_pairs: Iterable[tuple[str, str, int]]) -> None:
        """`train_pairs` yields (uid_i, uid_j, z_ij)."""
        self.counts = defaultdict(int)
        for _, uid_j, z in train_pairs:
            if z == 1:
                self.counts[uid_j] += 1

    def tune_gamma(
        self,
        train_pairs: list[tuple[str, str, int]],
        min_g: int = 1,
        max_g: int = 200,
    ) -> int:
        """Choose γ to maximize F1 on D_R^Train (per §3.1)."""
        from forecasting.eval_loop import f1_score

        # rebuild counts on the same training pairs
        self.fit(train_pairs)
        best_g, best_f1 = min_g, -1.0
        for g in range(min_g, max_g + 1):
            preds = [int(self.counts.get(uid_j, 0) >= g) for _, uid_j, _ in train_pairs]
            ys = [z for _, _, z in train_pairs]
            f1, _, _ = f1_score(preds, ys)
            if f1 > best_f1:
                best_f1, best_g = f1, g
        self.gamma = best_g
        return best_g

    # ----- inference ------------------------------------------------
    def predict(self, uid_j: str) -> int:
        return int(self.counts.get(uid_j, 0) >= self.gamma)

    def __call__(self, uid_i: str, uid_j: str) -> int:
        return self.predict(uid_j)
