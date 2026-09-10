"""Variance-Exploding (VE) and Variance-Preserving (VP) SDEs.

Implements the forward noising processes used in NPSE / TSNPSE
(Sharrock et al., ICML 2024, Appendix E.3.1), following Song et al., ICLR 2021.

Forward SDE (Eq. 2 of the paper):

    dθ_t = f(θ_t, t) dt + g(t) dw_t,    t ∈ (0, 1].

The two choices considered:

* VE SDE (Variance Exploding):
      f(θ, t) = 0
      g(t)   = σ_min · (σ_max / σ_min)^t · sqrt(2 log(σ_max / σ_min))
      transition density:
          p_{t|0}(θ_t | θ_0) = N(θ_t | θ_0,  σ_min² (σ_max/σ_min)^{2t} · I)
      σ_min = 0.01 for 2-D experiments (SIR, Two Moons),  0.05 otherwise.
      σ_max chosen by Technique 1 of Song & Ermon (2020): the maximum pairwise
      Euclidean distance between training data points.

* VP SDE (Variance Preserving):
      f(θ, t) = -½ β(t) θ
      g(t)    = sqrt(β(t))         with β(t) = β_min + t (β_max - β_min)
      transition density:
          p_{t|0}(θ_t | θ_0) = N( θ_t | θ_0 · α_t,  (1 - α_t²) · I )
      where α_t = exp(-½ ∫_0^t β_s ds).
      β_min = 0.1, β_max = 11.0 (paper Appendix E.3.1).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn


class BaseSDE(nn.Module):
    """Abstract base for forward / reverse SDE machinery used by the score model."""

    def __init__(self, T: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.T = float(T)
        self.eps = float(eps)  # avoid singularities at t = 0

    # --- Forward-process quantities ---------------------------------------
    def drift(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def marginal_prob(
        self, theta_0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std) of p_{t|0}(theta_t | theta_0), each broadcastable to theta_0."""
        raise NotImplementedError

    def prior_sampling(self, shape, device=None) -> torch.Tensor:
        """Sample from the stationary distribution π."""
        raise NotImplementedError

    # --- Convenience helpers ---------------------------------------------
    def add_noise(
        self, theta_0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample θ_t ~ p_{t|0}(·|θ_0) and return (θ_t, score_target, std).

        ``score_target`` is the denoising target ``∇_{θ_t} log p_{t|0}(θ_t | θ_0)``,
        which under a Gaussian transition with mean ``μ_t`` and std ``σ_t`` equals
        ``-(θ_t - μ_t) / σ_t²``.
        """
        mean, std = self.marginal_prob(theta_0, t)
        z = torch.randn_like(theta_0)
        theta_t = mean + std * z
        score_target = -(theta_t - mean) / (std**2)
        return theta_t, score_target, std

    # --- Reverse-time probability-flow ODE --------------------------------
    def probability_flow_drift(
        self,
        theta_t: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
    ) -> torch.Tensor:
        """Drift of the probability-flow ODE (Eq. 4 of the paper):
        dθ_t/dt = f(θ_t, t) - ½ g²(t) ∇log p_t(θ_t|x).
        """
        g = self.diffusion(t)
        # broadcast g over feature dim
        if g.ndim == theta_t.ndim - 1:
            g = g.unsqueeze(-1)
        return self.drift(theta_t, t) - 0.5 * (g**2) * score

    def reverse_sde_drift(
        self,
        theta_t: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
    ) -> torch.Tensor:
        """Drift of the reverse-time SDE (Eq. 3 of the paper):
            dθ̄_t = [-f(θ̄_t, T-t) + g²(T-t) ∇log p_{T-t}(θ̄_t|x)] dt + g(T-t) dw_t.

        For convenience here we return the *forward-time* reverse drift
        f(θ_t, t) - g²(t) ∇log p_t(θ_t | x), to be integrated from t = T → 0.
        """
        g = self.diffusion(t)
        if g.ndim == theta_t.ndim - 1:
            g = g.unsqueeze(-1)
        return self.drift(theta_t, t) - (g**2) * score


class VESDE(BaseSDE):
    """Variance Exploding SDE (Song et al., 2021; paper Appendix E.3.1)."""

    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        T: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__(T=T, eps=eps)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    @torch.no_grad()
    def set_sigma_max(self, sigma_max: float) -> None:
        """Update σ_max (e.g., chosen via Technique 1 of Song & Ermon 2020).

        Per the addendum: when computing σ_max for VE SDE in sequential methods,
        only the training data points available in the **first round** should
        be used.
        """
        self.sigma_max = float(max(sigma_max, self.sigma_min * 1.01))

    def drift(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(theta)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        ratio = self.sigma_max / self.sigma_min
        # g(t) = σ_min · ratio^t · sqrt(2 log ratio)
        return self.sigma_min * (ratio**t) * math.sqrt(2.0 * math.log(ratio))

    def marginal_std(self, t: torch.Tensor) -> torch.Tensor:
        # σ(t) = σ_min · (σ_max / σ_min)^t
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def marginal_prob(
        self, theta_0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        std = self.marginal_std(t)
        # Reshape for broadcasting (B,) -> (B, 1) so std broadcasts over feature dim.
        if std.ndim == theta_0.ndim - 1:
            std = std.unsqueeze(-1)
        mean = theta_0
        return mean, std

    def prior_sampling(self, shape, device=None) -> torch.Tensor:
        return torch.randn(*shape, device=device) * self.sigma_max


class VPSDE(BaseSDE):
    """Variance Preserving SDE (Song et al., 2021; paper Appendix E.3.1).

    β_min = 0.1, β_max = 11.0 by default (paper Appendix E.3.1).
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 11.0,
        T: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__(T=T, eps=eps)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def drift(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b = self.beta(t)
        if b.ndim == theta.ndim - 1:
            b = b.unsqueeze(-1)
        return -0.5 * b * theta

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.beta(t))

    def _log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        # ∫_0^t β_s ds = β_min t + ½ (β_max - β_min) t²
        return -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min

    def marginal_prob(
        self, theta_0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_mean = self._log_mean_coeff(t)
        if log_mean.ndim == theta_0.ndim - 1:
            log_mean = log_mean.unsqueeze(-1)
        alpha = torch.exp(log_mean)
        mean = alpha * theta_0
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean))
        return mean, std

    def prior_sampling(self, shape, device=None) -> torch.Tensor:
        # Stationary distribution of the VP SDE is N(0, I).
        return torch.randn(*shape, device=device)
