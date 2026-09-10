"""§3.2 Trainable Logit-Based Forecasting (Eqn. 2 + Eqn. 3).

Mathematical core (paper Eqn. 2):

    f̂_i(x_j)  =  Θ̃(x_j, x_i) · [ f̂_i(x_i) - f̂_0(x_i) ]   +   f̂_0(x_j)

with the trainable kernel:

    Θ̃(x_j, x_i)  =  h(x_j, y_j) · h(x_i, y_i)^T  ∈ ℝ^{T × T}

where h: (x, y) ↦ ℝ^{T × d}.

We then optimize the margin loss in Eqn. 3 on the predicted logits.

Efficient inference (§3.2 last paragraph):
    - f̂_0(x_j) and Δf̂_i(x_i) are cached once per (j) and (i) respectively.
    - Only the top-k = 100 largest logits per output token are stored.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None

from .encoder import Encoder, EncoderConfig, build_encoder
from .losses import margin_loss


# ---------------------------------------------------------------------
@dataclass
class LogitForecasterConfig:
    proj_dim: int = 256
    margin: float = 1.0
    top_k_logits: int = 100  # §3.2 efficient inference: cache top-k
    max_output_len: int = 64


class LogitForecaster(nn.Module if nn is not None else object):
    """Trainable logit-change forecaster (§3.2).

    The encoder h is shared across (x_i, y_i) and (x_j, y_j); the forecast
    is the kernel-projected logit change of x_i added to f_0's logits on x_j.
    """

    def __init__(
        self,
        cfg: LogitForecasterConfig | None = None,
        encoder_cfg: EncoderConfig | None = None,
    ):
        if nn is None:
            raise RuntimeError("PyTorch not installed")
        super().__init__()
        self.cfg = cfg or LogitForecasterConfig()
        self.encoder: Encoder = build_encoder(encoder_cfg)

    # -----------------------------------------------------------------
    def kernel(self, x_j: str, y_j: str, x_i: str, y_i: str):
        """Trainable kernel Θ̃(x_j, x_i) ∈ R^{T × T}."""
        h_j = self.encoder(x_j, y_j, pool="token")  # (T, d)
        h_i = self.encoder(x_i, y_i, pool="token")  # (T, d)
        return h_j @ h_i.t()  # (T_j, T_i)

    # -----------------------------------------------------------------
    def forecast_logits(
        self,
        x_j: str,
        y_j: str,
        x_i: str,
        y_i: str,
        delta_logits_xi,  # (T, V) = f̂_i(x_i) - f̂_0(x_i)
        f0_logits_xj,  # (T, V) = f̂_0(x_j)
    ):
        """Return the predicted logits f̂_i(x_j), Eqn. 2."""
        theta = self.kernel(x_j, y_j, x_i, y_i)  # (T_j, T_i)
        # T may differ; align to common dim
        T_j = min(theta.size(0), f0_logits_xj.size(0))
        T_i = min(theta.size(1), delta_logits_xi.size(0))
        theta = theta[:T_j, :T_i]
        delta = delta_logits_xi[:T_i]
        # propagate logit change: (T_j, T_i) @ (T_i, V) -> (T_j, V)
        propagated = theta @ delta
        forecast = f0_logits_xj[:T_j] + propagated
        return forecast

    # -----------------------------------------------------------------
    def predict_label(
        self,
        forecast_logits,
        y_j_token_ids,
    ) -> int:
        """Binary forecast: 1 iff predicted top-1 token != y_j (forgotten)."""
        T = min(forecast_logits.size(0), y_j_token_ids.size(0))
        pred = forecast_logits[:T].argmax(dim=-1)
        return int((pred != y_j_token_ids[:T]).any())

    def loss(
        self,
        forecast_logits,
        y_j_token_ids,
        z_ij: int,
    ):
        return margin_loss(
            forecast_logits, y_j_token_ids, z_ij=z_ij, margin=self.cfg.margin
        )


# ---------------------------------------------------------------------
class FixedLogitForecaster:
    """Non-trained variant of §3.2 (Table 1 'Fixed Logit' row).

    "Replaces trainable encoding function h with the frozen final-layer
     representation of the base PTLM.  The resulting kernel is identical
     to the ground-truth kernel when only the final LM heads are tuned."
    """

    def __init__(self, base_lm):
        self.base_lm = base_lm

    @torch.no_grad() if torch is not None else (lambda f: f)
    def _frozen_rep(self, x: str) -> "torch.Tensor":
        device = next(self.base_lm.model.parameters()).device
        enc = self.base_lm.tokenizer(
            x,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)
        encoder = getattr(
            self.base_lm.model, "encoder", getattr(self.base_lm.model, "model", None)
        )
        out = encoder(**enc, return_dict=True)
        return out.last_hidden_state[0]  # (L, H)

    def kernel(self, x_j: str, x_i: str):
        h_j = self._frozen_rep(x_j)
        h_i = self._frozen_rep(x_i)
        return h_j @ h_i.t()

    def forecast_logits(self, x_j, y_j, x_i, y_i, delta_logits_xi, f0_logits_xj):
        K = self.kernel(x_j, x_i)
        T_j = min(K.size(0), f0_logits_xj.size(0))
        T_i = min(K.size(1), delta_logits_xi.size(0))
        return f0_logits_xj[:T_j] + K[:T_j, :T_i] @ delta_logits_xi[:T_i]


# ---------------------------------------------------------------------
def _cache_logits(base_lm, examples, top_k: int = 100, max_T: int = 64):
    """Pre-compute and cache top-k logits per output token for each example.

    Mirrors the §3.2 efficient-inference description:
        "We only cache top k=100 largest logits for each token in y_j."
    """
    if torch is None:
        raise RuntimeError("PyTorch not installed")
    cache: dict[str, "torch.Tensor"] = {}
    base_lm.model.eval()
    device = next(base_lm.model.parameters()).device
    for ex in examples:
        enc = base_lm.tokenizer(
            ex.x,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            # Force-decode the gold y so logits align with token positions
            labels = base_lm.tokenizer(
                ex.y,
                return_tensors="pt",
                truncation=True,
                max_length=max_T,
            ).input_ids.to(device)
            out = base_lm.model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                labels=labels,
            )
        logits = out.logits[0]  # (T, V)
        values, idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
        cache[ex.uid] = (values.cpu(), idx.cpu())
    return cache
