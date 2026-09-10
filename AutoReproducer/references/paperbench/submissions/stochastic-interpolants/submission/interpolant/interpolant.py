"""The stochastic interpolant process I_t and its time derivative.

Implements Definition 3.1:

    I_t = α_t x_0 + β_t x_1 + γ_t z,
    İ_t = α̇_t x_0 + β̇_t x_1 + γ̇_t z,

where the pair (x_0, x_1) is sampled from the joint coupling
ρ(x_0, x_1 | ξ) (cf. §4.1, §4.2 and Definition A.1) and z ~ N(0, Id).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .coefficients import InterpolantCoefficients


def _broadcast(tensor: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Right-broadcast a (B,) scalar field to (B, 1, 1, ..., 1) image-shape."""
    while tensor.dim() < like.dim():
        tensor = tensor.unsqueeze(-1)
    return tensor


@dataclass
class StochasticInterpolant:
    """Container that builds I_t and İ_t from (x_0, x_1, z, t).

    Attributes
    ----------
    coeffs:
        An :class:`InterpolantCoefficients` instance providing α, β, γ
        and their derivatives.
    """

    coeffs: InterpolantCoefficients

    # ------------------------------------------------------------------
    # Forward construction (training-time helper).
    # ------------------------------------------------------------------
    def build(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (I_t, İ_t, z).

        Parameters
        ----------
        x0, x1: tensors with the same shape (B, ...). The base / target.
        t: scalar tensor of shape (B,) with values in [0, 1].
        z: optional Gaussian noise N(0, Id) of the same shape as x0; if
           None, drawn internally with the same dtype/device as x1.
        """
        if z is None:
            z = torch.randn_like(x1)

        a, b, g, ad, bd, gd = self.coeffs.all_coeffs(t)
        a = _broadcast(a, x1)
        b = _broadcast(b, x1)
        g = _broadcast(g, x1)
        ad = _broadcast(ad, x1)
        bd = _broadcast(bd, x1)
        gd = _broadcast(gd, x1)

        # Definition 3.1 — interpolant value
        It = a * x0 + b * x1 + g * z
        # Time-derivative used inside the velocity loss (§3.4 / Eq. 22)
        It_dot = ad * x0 + bd * x1 + gd * z
        return It, It_dot, z

    # ------------------------------------------------------------------
    # Helpers for the score / SDE machinery (Eq. 6 + Corollary 3.1).
    # ------------------------------------------------------------------
    def score_from_g(
        self, g_pred: torch.Tensor, t: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """∇log ρ_t(x) = -γ_t⁻¹ g_t(x) — Eq. (6)."""
        gamma = _broadcast(self.coeffs.gamma(t), g_pred)
        return -g_pred / torch.clamp(gamma, min=eps)
