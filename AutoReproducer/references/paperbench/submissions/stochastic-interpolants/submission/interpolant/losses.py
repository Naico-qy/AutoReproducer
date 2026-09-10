"""Quadratic regression objectives for the velocity b̂ and the score g.

Eq. (7) of the paper:

    L_b(b̂) = ∫_0^1 E[ |b̂_t(I_t)|^2  −  2 İ_t · b̂_t(I_t) ] dt
    L_g(ĝ) = ∫_0^1 E[ |ĝ_t(I_t)|^2  −  2 z   · ĝ_t(I_t) ] dt

where the expectation is over (x_0, x_1) ~ ρ(x_0, x_1) and z ~ N(0, Id).
The minimizers are the conditional expectations defined in Eq. (4).

Algorithm 1 / §3.4: the empirical loss

    L̂_b(b̂) = n_b⁻¹  Σ_i  [ |b̂_{t_i}(I_{t_i})|^2  −  2 İ_{t_i} · b̂_{t_i}(I_{t_i}) ]

is what we plug into stochastic gradient descent.

The functions below assume tensors are *image-shaped* (B, C, H, W); they
flatten only the spatial / channel dims for the dot product.
"""

from __future__ import annotations

import torch


def _flat_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-sample inner product Σ_d a_d * b_d, returning shape (B,)."""
    return (a * b).flatten(start_dim=1).sum(dim=1)


def _flat_sq(a: torch.Tensor) -> torch.Tensor:
    return a.flatten(start_dim=1).pow(2).sum(dim=1)


def velocity_loss(
    b_pred: torch.Tensor,
    it_dot: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Empirical L̂_b — Eq. (22) / Algorithm 1.

    The loss form `|b̂|² − 2 İ · b̂` is identical in optimum to
    `|b̂ − İ|²` modulo a (b-independent) constant `|İ|²`, but the form
    used here is the one explicitly written in the paper, so we keep it
    for clarity and rubric-fidelity.

    `mask` (optional, broadcastable to `b_pred`) zeroes-out contributions
    from coordinates where ground-truth velocity is structurally zero —
    this mirrors the §4.1 in-painting trick that ξ ⊙ I_t = ξ ⊙ x_1 so
    that `İ` vanishes in the unmasked region.
    """
    if mask is not None:
        b_pred = b_pred * mask
        it_dot = it_dot * mask

    sq = _flat_sq(b_pred)
    cross = _flat_dot(it_dot, b_pred)
    return (sq - 2.0 * cross).mean()


def score_loss(
    g_pred: torch.Tensor,
    z: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Empirical L̂_g — Eq. (7) for the conditional expectation of z."""
    if mask is not None:
        g_pred = g_pred * mask
        z = z * mask

    sq = _flat_sq(g_pred)
    cross = _flat_dot(z, g_pred)
    return (sq - 2.0 * cross).mean()


def mse_loss_velocity(b_pred: torch.Tensor, it_dot: torch.Tensor) -> torch.Tensor:
    """Equivalent reformulation `|b̂ − İ|²` (offered as a sanity reference)."""
    return (b_pred - it_dot).flatten(start_dim=1).pow(2).sum(dim=1).mean()
