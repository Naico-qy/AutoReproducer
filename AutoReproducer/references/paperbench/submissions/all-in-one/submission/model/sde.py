"""Stochastic differential equations for the Simformer.

Implements VESDE and VPSDE following Song et al. (2021),
"Score-based generative modeling through stochastic differential equations."
The paper §2.3 and §4.1 use VESDE for the main results; VPSDE results are in
Appendix A3. The default config selects VESDE.

For SBI we operate on the joint x_hat = (theta, x).
"""

from __future__ import annotations

import math
import torch
from torch import Tensor


class BaseSDE:
    """Abstract SDE: dx_t = f(x,t) dt + g(t) dw."""

    T: float = 1.0
    eps: float = 1e-3

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError

    def diffusion(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def marginal(self, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Return mean and std of p_t(x_t | x_0) (Gaussian by construction)."""
        raise NotImplementedError

    def prior_sample(self, shape, device) -> Tensor:
        raise NotImplementedError

    def perturb(self, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Sample x_t ~ p_t(.|x_0); return (x_t, mean, std).

        The score of the conditional Gaussian transition is
        ∇_{x_t} log p_t(x_t | x_0) = -(x_t - mean) / std**2.
        """
        mean, std = self.marginal(x0, t)
        noise = torch.randn_like(x0)
        # Broadcast std over feature dims
        while std.dim() < x0.dim():
            std = std.unsqueeze(-1)
            mean = mean if mean.dim() == x0.dim() else mean.unsqueeze(-1)
        x_t = mean + std * noise
        return x_t, mean, std


class VESDE(BaseSDE):
    """Variance Exploding SDE  (Song et al. 2021).

    dx = sqrt(d sigma^2(t)/dt) dw,  with sigma(t) = sigma_min * (sigma_max/sigma_min)^t.
    Drift f = 0; diffusion g(t) follows from chain rule.
    """

    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 15.0,
        T: float = 1.0,
        eps: float = 1e-3,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.T = T
        self.eps = eps
        self._log_ratio = math.log(self.sigma_max / self.sigma_min)

    def sigma(self, t: Tensor) -> Tensor:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        return torch.zeros_like(x)

    def diffusion(self, t: Tensor) -> Tensor:
        # g(t) = sigma(t) * sqrt(2 log(sigma_max/sigma_min))
        return self.sigma(t) * math.sqrt(2.0 * self._log_ratio)

    def marginal(self, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        mean = x0
        std = self.sigma(t)
        return mean, std

    def prior_sample(self, shape, device) -> Tensor:
        return torch.randn(*shape, device=device) * self.sigma_max


class VPSDE(BaseSDE):
    """Variance Preserving SDE  (Song et al. 2021).

    dx = -1/2 beta(t) x dt + sqrt(beta(t)) dw,  beta linear in t.
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        T: float = 1.0,
        eps: float = 1e-3,
    ) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.eps = eps

    def beta(self, t: Tensor) -> Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        b = self.beta(t)
        while b.dim() < x.dim():
            b = b.unsqueeze(-1)
        return -0.5 * b * x

    def diffusion(self, t: Tensor) -> Tensor:
        return torch.sqrt(self.beta(t))

    def marginal(self, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        # log_alpha_t = -1/4 t^2 (bmax - bmin) - 1/2 t bmin
        log_alpha = (
            -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )
        alpha = torch.exp(log_alpha)
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_alpha))
        # Mean = alpha * x0 (broadcast)
        a = alpha
        while a.dim() < x0.dim():
            a = a.unsqueeze(-1)
        mean = a * x0
        return mean, std

    def prior_sample(self, shape, device) -> Tensor:
        return torch.randn(*shape, device=device)


def get_sde(cfg) -> BaseSDE:
    """Factory from the YAML config block ``sde``."""
    kind = cfg.get("type", "vesde").lower()
    if kind == "vesde":
        return VESDE(
            sigma_min=cfg.get("sigma_min", 0.01),
            sigma_max=cfg.get("sigma_max", 15.0),
            T=cfg.get("T", 1.0),
            eps=cfg.get("eps", 1e-3),
        )
    if kind == "vpsde":
        return VPSDE(
            beta_min=cfg.get("beta_min", 0.1),
            beta_max=cfg.get("beta_max", 20.0),
            T=cfg.get("T", 1.0),
            eps=cfg.get("eps", 1e-3),
        )
    raise ValueError(f"Unknown SDE type: {kind}")
