"""Gaussian diffusion schedule and forward / reverse process helpers.

Reference  : Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
             (CrossRef-verified citation: Ojha et al., CVPR 2021,
              DOI: 10.1109/CVPR46437.2021.01060 -- the canonical few-shot
              image-generation benchmark this paper compares against.)

Implements paper §3 Equations (1), (2), (3) of Wang et al. ICML 2024
"Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based
Transfer Learning":

    q(x_t | x_0)        = N(x_t;  sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)
    x_t                 = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps
    L_simple(theta)     = E[ ||eps - eps_theta(x_t, t)||^2 ]
    sigma_hat_t         = (1 - alpha_bar_{t-1}) * sqrt(alpha_t / (1 - alpha_bar_t))
"""

from __future__ import annotations

import math
import torch
from torch import Tensor


def make_beta_schedule(
    name: str, T: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> Tensor:
    """Construct β schedule. Supports the linear schedule of Ho et al. 2020
    and the cosine schedule of Nichol & Dhariwal 2021."""
    if name == "linear":
        return torch.linspace(beta_start, beta_end, T)
    if name == "cosine":
        s = 0.008
        steps = T + 1
        x = torch.linspace(0, T, steps)
        ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        ac = ac / ac[0]
        betas = 1 - (ac[1:] / ac[:-1])
        return betas.clamp(max=0.999)
    raise ValueError(f"unknown β schedule {name}")


class GaussianDiffusion:
    """Discrete-time Gaussian diffusion in the variance-preserving formulation.

    Stores the schedule constants (β_t, α_t, ᾱ_t) and provides q-sample,
    posterior moments, σ̂_t (paper Eq. 5 normalization) and a DDIM reverse
    step (Song et al. 2020, η = 0)."""

    def __init__(
        self,
        T: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device | str = "cpu",
    ) -> None:
        self.T = T
        betas = make_beta_schedule(beta_schedule, T, beta_start, beta_end).to(device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1, device=device), alpha_bar[:-1]])

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.alpha_bar_prev = alpha_bar_prev
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
        # σ̂_t = (1 - ᾱ_{t-1}) * sqrt(α_t / (1 - ᾱ_t))   (paper Eq. (5))
        self.sigma_hat = (1.0 - alpha_bar_prev) * torch.sqrt(alphas / (1.0 - alpha_bar))

    # ---------------------------------------------------------------
    # Forward process
    # ---------------------------------------------------------------
    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        """x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε  (paper Eq. (1))."""
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self._gather(self.sqrt_alpha_bar, t, x0.shape)
        som = self._gather(self.sqrt_one_minus_alpha_bar, t, x0.shape)
        return sa * x0 + som * noise

    # ---------------------------------------------------------------
    # Reverse process (DDIM, η = 0)
    # ---------------------------------------------------------------
    def predict_x0_from_eps(self, x_t: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        sa = self._gather(self.sqrt_alpha_bar, t, x_t.shape)
        som = self._gather(self.sqrt_one_minus_alpha_bar, t, x_t.shape)
        return (x_t - som * eps) / sa.clamp(min=1e-8)

    def ddim_step(self, x_t: Tensor, t: Tensor, t_prev: Tensor, eps: Tensor) -> Tensor:
        """Deterministic DDIM step (Song et al. 2020, η=0)."""
        ab_t = self._gather(self.alpha_bar, t, x_t.shape)
        ab_prev = self._gather(self.alpha_bar, t_prev, x_t.shape).clamp(min=1e-8)
        x0_pred = (x_t - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t.clamp(min=1e-8))
        dir_xt = torch.sqrt(1 - ab_prev) * eps
        return torch.sqrt(ab_prev) * x0_pred + dir_xt

    def get_sigma_hat(self, t: Tensor, ref: Tensor) -> Tensor:
        return self._gather(self.sigma_hat, t, ref.shape)

    @staticmethod
    def _gather(values: Tensor, t: Tensor, shape) -> Tensor:
        out = values.gather(0, t)
        return out.view(t.shape[0], *([1] * (len(shape) - 1)))
