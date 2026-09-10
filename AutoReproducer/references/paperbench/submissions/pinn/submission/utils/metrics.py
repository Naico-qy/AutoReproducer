"""L2 relative error and gradient norm utilities (Section 2.2)."""

from __future__ import annotations

import torch


def l2_relative_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """L2RE = sqrt( Σ (y - y')² / Σ y'² )  (Eq. between (2) and (3))."""
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    num = torch.sum((pred - target) ** 2)
    den = torch.sum(target**2).clamp_min(1e-30)
    return float(torch.sqrt(num / den))


def gradient_norm(loss: torch.Tensor, params) -> float:
    """Return ||∇L(w)||₂ for the given loss / parameter set."""
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    sq = 0.0
    for g in grads:
        if g is not None:
            sq += float(torch.sum(g**2))
    return float(sq**0.5)
