"""Loss functions used by the §3.2 / §3.3 forecasters.

Eqn. 3 (logit-based forecaster) — margin loss:

    L = max(0,
            1 + (-1)^{z_{ij}} · ( max_{v ≠ y_j} f̂_i(x_j)[v] - f̂_i(x_j)[y_j] ))

Eqn. 4 (representation-based forecaster) — binary cross entropy on
σ( h(x_j, y_j) · h(x_i, y_i)^T   +   b_j ).
"""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None


# ---------------------------------------------------------------------
def margin_loss(
    forecast_logits,  # (T, V) predicted logits f̂_i(x_j) for x_j
    y_j_token_ids,  # (T,)   correct token ids of y_j
    z_ij: int,  # 0 = not forgotten, 1 = forgotten
    margin: float = 1.0,
):
    """Margin loss from Eqn. 3 of the paper.

    The intent: if z_ij == 0 (NOT forgotten) we want the gold-token logit
    to exceed the second-best by at least `margin`. If z_ij == 1
    (forgotten) we want the inequality reversed.

    We average over output tokens T.
    """
    if torch is None:
        raise RuntimeError("PyTorch not installed")

    # gold logits for each output position: (T,)
    T, V = forecast_logits.shape
    gold = forecast_logits[torch.arange(T), y_j_token_ids]  # (T,)

    # second-top candidate per position: mask out the gold and take max
    masked = forecast_logits.clone()
    masked[torch.arange(T), y_j_token_ids] = float("-inf")
    second = masked.max(dim=-1).values  # (T,)

    sign = (-1.0) ** float(z_ij)
    losses = torch.clamp(margin + sign * (second - gold), min=0.0)
    return losses.mean()


# ---------------------------------------------------------------------
def binary_ce_loss(logits, target):
    """Binary cross-entropy with logits — used by the §3.3 forecaster."""
    if F is None:
        raise RuntimeError("PyTorch not installed")
    if not torch.is_tensor(target):
        target = torch.as_tensor(target, dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits.view(-1), target.view(-1).float())
