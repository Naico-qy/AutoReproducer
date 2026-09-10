"""Binary noise-image classifier p_phi(y | x_t) used for similarity-guided
training (paper §4.1, Eq. 4-5).

Design choices (from addendum.md):
  * Backbone   : pre-trained ADM classifier
                   - DDPM (256x256): https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt
                   - LDM   (64x64) : https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt
  * Head       : modify the final layer to output 2 logits
                 (source vs. target)
  * Optimizer  : Adam,  lr = 1e-4,  batch = 64,  iters = 300
  * Input      : noised image x_t at a uniformly-sampled timestep t,
                 i.e. trained on the same forward-noising distribution
                 q(x_t | x_0) used by the diffusion model.

The gradient that enters Eq. (5) is
    grad_x log p_phi(y = T | x_t)
where  T  is the integer label of the target domain (= 1 here, with
0 = source). The helper `similarity_grad()` returns this signal.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .unet import SinusoidalTimeEmbedding, ResBlock, AttentionBlock, Downsample, _norm


class BinaryNoiseClassifier(nn.Module):
    """Time-conditioned binary classifier on noised images x_t.

    Architecturally this is a slim ADM encoder (the encoder half of the
    classifier guidance network) followed by global pooling and a 2-way
    linear head. It is small enough to train from scratch on the
    10-image target set in 300 iterations (addendum), but it is
    designed to load the ADM classifier checkpoint when available.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions=(16, 8),
        image_size: int = 256,
        num_heads: int = 4,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        t_dim = base_channels * 4
        self.t_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        blocks: list[nn.Module] = []
        ch_in = base_channels
        cur_res = image_size
        for i, mult in enumerate(channel_mult):
            ch_out = base_channels * mult
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch_in, ch_out, t_dim))
                if cur_res in attention_resolutions:
                    blocks.append(AttentionBlock(ch_out, num_heads))
                ch_in = ch_out
            if i != len(channel_mult) - 1:
                blocks.append(Downsample(ch_in))
                cur_res //= 2
        self.blocks = nn.ModuleList(blocks)

        self.norm = _norm(ch_in)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(ch_in, num_classes)

    # ---------------------------------------------------------------
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Return logits  shape [B, num_classes]."""
        t_emb = self.t_embed(t)
        h = self.in_conv(x)
        for layer in self.blocks:
            h = layer(h, t_emb) if isinstance(layer, ResBlock) else layer(h)
        h = F.silu(self.norm(h))
        h = self.pool(h).flatten(1)
        return self.head(h)

    # ---------------------------------------------------------------
    def log_prob(self, x: Tensor, t: Tensor, y: int | Tensor) -> Tensor:
        """log p_phi(y | x_t)."""
        logits = self.forward(x, t)
        log_prob = F.log_softmax(logits, dim=-1)
        if isinstance(y, int):
            y = torch.full((x.size(0),), y, device=x.device, dtype=torch.long)
        return log_prob.gather(1, y[:, None]).squeeze(1)


# ---------------------------------------------------------------------
# Similarity gradient ∇_{x_t} log p_phi(y = T | x_t)   (paper Eq. 5)
# ---------------------------------------------------------------------
def similarity_grad(
    classifier: BinaryNoiseClassifier, x_t: Tensor, t: Tensor, target_label: int = 1
) -> Tensor:
    """Compute ∇_{x_t} log p_phi(y = target | x_t).

    This is the classifier-guidance gradient of Dhariwal & Nichol 2021,
    used in paper Eq. (5) as the similarity term that bridges the
    source-target domain gap during transfer learning.
    """
    x_t = x_t.detach().requires_grad_(True)
    log_p = classifier.log_prob(x_t, t, target_label).sum()
    grad = torch.autograd.grad(log_p, x_t, create_graph=False)[0]
    return grad.detach()
