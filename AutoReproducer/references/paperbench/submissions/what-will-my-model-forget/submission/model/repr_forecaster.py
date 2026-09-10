"""§3.3 Representation-Based Forecasting (Eqn. 4 + Frequency Prior).

    g((x_i, y_i), (x_j, y_j)) = σ( h(x_j, y_j) · h(x_i, y_i)^T  +  b_j )

where h: (x, y) ↦ ℝ^d is the *averaged* token representation of the
trainable encoder, and b_j is the log-odds frequency prior:

    b_j = log( |{ i : z_{ij} = 1 }| / |D_R^Train| )
        - log( |{ i : z_{ij} = 0 }| / |D_R^Train| )

The frequency-prior bias forces the representation to fit the *residual*
beyond what threshold-based forecasting captures.

Trained with binary cross-entropy.  Inference: σ(...) > 0.5 → forgotten.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

from .encoder import Encoder, EncoderConfig, build_encoder


# ---------------------------------------------------------------------
def frequency_prior(
    train_pairs: Iterable[tuple[str, str, int]],
) -> dict[str, float]:
    """Compute b_j for every upstream uid_j seen in train_pairs.

    Per §3.3:  b_j = log(p_forgotten) - log(p_not).
    """
    counts_pos: Counter[str] = Counter()
    counts_total = 0
    counts_neg: Counter[str] = Counter()
    for _, uid_j, z in train_pairs:
        counts_total += 1
        if z == 1:
            counts_pos[uid_j] += 1
        else:
            counts_neg[uid_j] += 1

    b: dict[str, float] = {}
    for uid_j in set(list(counts_pos) + list(counts_neg)):
        p = (counts_pos[uid_j] + 1e-6) / max(counts_total, 1)
        q = (counts_neg[uid_j] + 1e-6) / max(counts_total, 1)
        b[uid_j] = math.log(p) - math.log(q)
    return b


# ---------------------------------------------------------------------
@dataclass
class ReprForecasterConfig:
    proj_dim: int = 256
    use_frequency_prior: bool = True


class RepresentationForecaster(nn.Module if nn is not None else object):
    """Black-box dot-product forecaster (§3.3, Eqn. 4)."""

    def __init__(
        self,
        cfg: ReprForecasterConfig | None = None,
        encoder_cfg: EncoderConfig | None = None,
    ):
        if nn is None:
            raise RuntimeError("PyTorch not installed")
        super().__init__()
        self.cfg = cfg or ReprForecasterConfig()
        self.encoder: Encoder = build_encoder(encoder_cfg)
        self._freq_bias: dict[str, float] = {}

    # -----------------------------------------------------------------
    def set_frequency_prior(self, b: dict[str, float]) -> None:
        self._freq_bias = b if self.cfg.use_frequency_prior else {}

    # -----------------------------------------------------------------
    def forward(
        self,
        x_i: str,
        y_i: str,
        x_j: str,
        y_j: str,
        uid_j: str | None = None,
    ):
        """Return the raw logit (pre-sigmoid) of forgetting."""
        h_i = self.encoder(x_i, y_i, pool="mean")  # (d,)
        h_j = self.encoder(x_j, y_j, pool="mean")  # (d,)
        score = (h_j * h_i).sum()  # scalar (dot product)
        if uid_j is not None and uid_j in self._freq_bias:
            score = score + self._freq_bias[uid_j]
        return score

    # -----------------------------------------------------------------
    @torch.no_grad() if torch is not None else (lambda f: f)
    def predict(self, x_i, y_i, x_j, y_j, uid_j=None) -> int:
        s = self.forward(x_i, y_i, x_j, y_j, uid_j=uid_j)
        return int(torch.sigmoid(s).item() > 0.5)
