"""Interpolant coefficient functions α_t, β_t, γ_t.

Definition 3.1 of Albergo et al. (ICML 2024) requires α, β, γ that are
differentiable on [0, 1] with the boundary conditions

    α_0 = β_1 = 1,
    α_1 = β_0 = γ_0 = γ_1 = 0,
    α_t² + β_t² + γ_t² > 0 ∀ t ∈ [0, 1].

The paper highlights two specific schedules:

    * Linear with noise (the "simple instance" from §3, page 3):
        α_t = 1 - t,  β_t = t,  γ_t = sqrt(2 t (1 - t))

    * Linear without noise (Eq. 20 / §4.1 / §4.2 — used in all experiments):
        α_t = t,  β_t = 1 - t,  γ_t = 0

Note: the §4.1 in-painting setting flips the convention with α_t = t,
β_t = 1-t.  We expose both via the `swap` flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import math
import torch


def _ensure_tensor(
    t: torch.Tensor | float, ref: torch.Tensor | None = None
) -> torch.Tensor:
    if isinstance(t, torch.Tensor):
        return t
    out = torch.tensor(t, dtype=torch.float32)
    if ref is not None:
        out = out.to(ref.device).to(ref.dtype)
    return out


class InterpolantCoefficients:
    """Abstract base class for (α, β, γ) and their time-derivatives."""

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def gamma_dot(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # -- convenience accessors --------------------------------------------
    def all_coeffs(self, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return (
            self.alpha(t),
            self.beta(t),
            self.gamma(t),
            self.alpha_dot(t),
            self.beta_dot(t),
            self.gamma_dot(t),
        )

    def has_noise(self) -> bool:
        return True


@dataclass
class LinearWithNoise(InterpolantCoefficients):
    """Definition 3.1 simple instance.

    α_t = 1 - t,  β_t = t,  γ_t = sqrt(2 t (1 - t)).
    """

    eps: float = 1e-6

    def alpha(self, t):
        return 1.0 - t

    def beta(self, t):
        return t

    def gamma(self, t):
        return torch.sqrt(torch.clamp(2.0 * t * (1.0 - t), min=self.eps))

    def alpha_dot(self, t):
        return -torch.ones_like(t)

    def beta_dot(self, t):
        return torch.ones_like(t)

    def gamma_dot(self, t):
        # d/dt sqrt(2t(1-t)) = (1-2t)/sqrt(2t(1-t))
        denom = torch.sqrt(torch.clamp(2.0 * t * (1.0 - t), min=self.eps))
        return (1.0 - 2.0 * t) / denom

    def has_noise(self) -> bool:
        return True


@dataclass
class LinearNoNoise(InterpolantCoefficients):
    """Eq. 20 of the paper — α_t = t, β_t = 1-t, γ_t = 0.

    Used in §4.1 in-painting and §4.2 super-resolution.
    With this choice, x_0 already contains paired information about x_1
    (e.g. an upsampled low-res image plus σ ζ), so the extra Gaussian
    γ_t z is unnecessary.
    """

    swap: bool = True  # if True follow §4.1 convention α_t=t, β_t=1-t

    def alpha(self, t):
        return t if self.swap else (1.0 - t)

    def beta(self, t):
        return (1.0 - t) if self.swap else t

    def gamma(self, t):
        return torch.zeros_like(t)

    def alpha_dot(self, t):
        return torch.ones_like(t) if self.swap else -torch.ones_like(t)

    def beta_dot(self, t):
        return -torch.ones_like(t) if self.swap else torch.ones_like(t)

    def gamma_dot(self, t):
        return torch.zeros_like(t)

    def has_noise(self) -> bool:
        return False


def make_coefficients(name: str) -> InterpolantCoefficients:
    """Factory used by configs."""
    name = name.lower()
    if name in ("linear", "linear_no_noise"):
        return LinearNoNoise(swap=True)
    if name in ("linear_with_noise", "trig_noise", "default"):
        return LinearWithNoise()
    raise ValueError(f"Unknown interpolant schedule: {name!r}")
