"""DPMs-ANT main module — implements Algorithm 1 of

  Wang, X., Lin, B., Liu, D., Chen, Y.-C., Xu, C.
  "Bridging Data Gaps in Diffusion Models with Adversarial
   Noise-Based Transfer Learning"
  ICML 2024.

This file ties together
  - U-Net   ε_θ(x_t, t)              (model/unet.py)
  - Adaptor ψ                         (model/adaptor.py)
  - Binary classifier p_phi(y|x_t)    (model/classifier.py)
  - Diffusion schedule                (model/schedule.py)
and exposes:

    DPMsANT.training_step(x0)  -> scalar loss L(ψ)         (Eq. 8)
    DPMsANT.adversarial_noise(x0, t)  -> ε* via Eq. (7)
    DPMsANT.similarity_loss(x0, t, eps) -> Eq. (5)
    DPMsANT.sample(N)          -> N images via DDIM
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .schedule import GaussianDiffusion
from .unet import UNet
from .adaptor import AdaptedUNet
from .classifier import BinaryNoiseClassifier, similarity_grad


# ---------------------------------------------------------------------
# Helper: Norm(·) operator from paper Eq. (7)
#   "approximately ensures the mean and standard deviation of ε^{j+1}
#    is 0 and I, respectively."
# ---------------------------------------------------------------------
def _normalize_noise(eps: Tensor, eps_=1e-5) -> Tensor:
    flat = eps.flatten(1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True) + eps_
    out = (flat - mean) / std
    return out.reshape_as(eps)


# ---------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------
@dataclass
class DPMsANTConfig:
    # diffusion
    T: int = 1000
    beta_schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    # adaptor
    adaptor_c: int = 4
    adaptor_d: int = 8
    adaptor_heads: int = 4
    adaptor_zero_init: bool = True
    freeze_backbone: bool = True
    # similarity guidance  (paper §4.1, Eq. 5)
    gamma: float = 5.0
    target_label: int = 1
    # adversarial noise selection  (paper §4.2, Eq. 7)
    use_adv_noise: bool = True
    adv_J: int = 10
    adv_omega: float = 0.02


class DPMsANT(nn.Module):
    def __init__(
        self,
        unet: UNet,
        classifier: BinaryNoiseClassifier,
        cfg: DPMsANTConfig,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)

        self.diffusion = GaussianDiffusion(
            T=cfg.T,
            beta_schedule=cfg.beta_schedule,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            device=self.device,
        )

        # Trainable adapted U-Net (θ frozen, ψ trainable)
        self.eps_model = AdaptedUNet(
            unet=unet,
            c=cfg.adaptor_c,
            d=cfg.adaptor_d,
            num_heads=cfg.adaptor_heads,
            zero_init=cfg.adaptor_zero_init,
            freeze_backbone=cfg.freeze_backbone,
        ).to(self.device)

        # Frozen pre-trained ε_θ used by Eq. (7) (adversarial inner max).
        # We re-use the same UNet weights but in eval/no-grad mode.
        self.frozen_eps = unet.to(self.device)
        for p in self.frozen_eps.parameters():
            p.requires_grad_(False)
        self.frozen_eps.eval()

        # Frozen classifier p_phi (paper Eq. 5)
        self.classifier = classifier.to(self.device).eval()
        for p in self.classifier.parameters():
            p.requires_grad_(False)

    # =================================================================
    # Eq. (7) — Adversarial Noise selection (multi-step PGD-like ascent)
    #
    #   ε^{j+1} = Norm( ε^j + ω · ∇_ε ||ε^j - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε^j, t)||² )
    #
    # The "similarity-guided term is disregarded, as this term is hard
    # to compute differential and is almost unchanged in the process"
    # (paper §4.2).
    # =================================================================
    @torch.enable_grad()
    def adversarial_noise(
        self, x0: Tensor, t: Tensor, eps0: Optional[Tensor] = None
    ) -> Tensor:
        cfg = self.cfg
        if eps0 is None:
            eps0 = torch.randn_like(x0)
        eps_j = eps0.detach()

        sa = self.diffusion._gather(self.diffusion.sqrt_alpha_bar, t, x0.shape)
        som = self.diffusion._gather(
            self.diffusion.sqrt_one_minus_alpha_bar, t, x0.shape
        )

        for _ in range(cfg.adv_J):
            eps_j = eps_j.detach().requires_grad_(True)
            x_t = sa * x0 + som * eps_j
            eps_pred = self.frozen_eps(x_t, t)
            loss = (eps_j - eps_pred).pow(2).mean()
            grad = torch.autograd.grad(loss, eps_j)[0]
            eps_next = eps_j.detach() + cfg.adv_omega * grad.detach()
            eps_j = _normalize_noise(eps_next)
        return eps_j.detach()

    # =================================================================
    # Eq. (5) — similarity-guided DPM training loss
    #
    #   L(ψ) = E[ ||ε - ε_{θ,ψ}(x_t, t) - σ̂_t² γ ∇_x log p_phi(y=T|x_t)||² ]
    # =================================================================
    def similarity_loss(self, x0: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        sa = self.diffusion._gather(self.diffusion.sqrt_alpha_bar, t, x0.shape)
        som = self.diffusion._gather(
            self.diffusion.sqrt_one_minus_alpha_bar, t, x0.shape
        )
        x_t = sa * x0 + som * eps

        eps_pred = self.eps_model(x_t, t)

        if self.cfg.gamma > 0:
            grad = similarity_grad(
                self.classifier, x_t, t, target_label=self.cfg.target_label
            )
            sigma_hat = self.diffusion.get_sigma_hat(t, x0)
            guidance = (sigma_hat**2) * self.cfg.gamma * grad
        else:
            guidance = torch.zeros_like(eps)

        residual = eps - eps_pred - guidance
        return residual.pow(2).mean()

    # =================================================================
    # Algorithm 1 -- one training step
    # =================================================================
    def training_step(self, x0: Tensor) -> Tensor:
        b = x0.size(0)
        t = torch.randint(0, self.cfg.T, (b,), device=self.device, dtype=torch.long)

        # Inner loop: adversarial noise selection (Eq. 7)
        if self.cfg.use_adv_noise:
            eps_star = self.adversarial_noise(x0, t)
        else:
            eps_star = torch.randn_like(x0)

        # Outer loop: minimize Eq. (8) wrt ψ
        return self.similarity_loss(x0, t, eps_star)

    # -----------------------------------------------------------------
    # Inference: DDIM sampling using the adapted model.
    # -----------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        n: int,
        image_size: int,
        channels: int = 3,
        steps: int = 50,
        eta: float = 0.0,
    ) -> Tensor:
        T = self.cfg.T
        ts = torch.linspace(T - 1, 0, steps + 1, device=self.device).long()
        x = torch.randn(n, channels, image_size, image_size, device=self.device)
        for i in range(steps):
            t_cur = ts[i].expand(n)
            t_prev = ts[i + 1].expand(n)
            eps = self.eps_model(x, t_cur)
            x = self.diffusion.ddim_step(x, t_cur, t_prev, eps)
        return x.clamp(-1, 1)

    # -----------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------
    def trainable_parameters(self):
        return self.eps_model.trainable_parameters()
