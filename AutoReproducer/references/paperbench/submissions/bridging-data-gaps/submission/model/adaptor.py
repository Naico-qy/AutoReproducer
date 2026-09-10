"""Adaptor module ψ inserted between U-Net blocks (paper §4.3).

Per addendum.md (verbatim):
    "The adaptor module is composed of a down-pooling layer followed by a
     normalization layer with 3x3 convolution. Then there is a 4 head
     attention layer followed by an MLP layer reducing feature size to
     8 or 16. Then there is an up-sampling layer with a factor of 4, a
     normalization layer, and 3x3 convolutions."

Connection rule (paper §4.3):
    x_t^l = θ^l(x_t^{l-1}) + ψ^l(x_t^{l-1})

Initialization: zero-init so that the adapted model exactly equals the
pre-trained source model at iteration 0 (paper §5.2: "we set all the
extra layer parameters to zero").
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .unet import UNet, _norm


class Adaptor(nn.Module):
    """ψ^l: per-block residual adaptor.

    Args:
        ch:        channel dim of the U-Net feature map at this depth
        c:         spatial down-projection factor   (paper: c=4 for DDPM, c=2 for LDM)
        d:         bottleneck channel dim           (paper: d=8)
        num_heads: attention heads                  (addendum: 4)
        zero_init: initialize the final conv to zero so ψ(x)=0 at start
    """

    def __init__(
        self,
        ch: int,
        c: int = 4,
        d: int = 8,
        num_heads: int = 4,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.ch = ch
        self.c = c
        self.d = d

        # 1) Down-pooling layer (avg-pool by factor c)
        self.down_pool = (
            nn.AvgPool2d(kernel_size=c, stride=c) if c > 1 else nn.Identity()
        )

        # 2) Normalization + 3x3 conv (project to bottleneck channels d)
        self.norm1 = _norm(ch)
        self.conv1 = nn.Conv2d(ch, d, kernel_size=3, padding=1)

        # 3) 4-head self-attention over the down-pooled spatial map
        attn_heads = num_heads if (d % num_heads == 0) else 1
        self.attn = _Attn(d, heads=attn_heads)

        # 4) MLP reducing feature size to d (already d -> identity-ish MLP)
        #    The addendum says "MLP layer reducing feature size to 8 or 16",
        #    which we interpret as a channel-mixing 1x1 MLP at bottleneck=d.
        self.mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d),
        )

        # 5) Up-sampling (factor 4 per addendum), normalization, 3x3 conv
        #    The combined down(c) + up(4) makes the adaptor act at a
        #    coarser scale; if c != 4 the residual is interpolated back
        #    to the input spatial size before adding.
        self.up_factor = 4
        self.norm2 = _norm(d)
        self.conv2 = nn.Conv2d(d, ch, kernel_size=3, padding=1)

        # zero-init final conv so ψ outputs 0 at start (paper §5.2)
        if zero_init:
            nn.init.zeros_(self.conv2.weight)
            nn.init.zeros_(self.conv2.bias)

    # ---------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        b, ch, h, w = x.shape

        z = self.down_pool(x)
        z = self.conv1(F.silu(self.norm1(z)))

        # spatial attention
        z = self.attn(z)

        # MLP over channel dim (treat each spatial loc as a token)
        zb, zc, zh, zw = z.shape
        zt = z.permute(0, 2, 3, 1).reshape(zb, zh * zw, zc)
        zt = zt + self.mlp(zt)
        z = zt.reshape(zb, zh, zw, zc).permute(0, 3, 1, 2)

        # up-sample by 4
        z = F.interpolate(z, scale_factor=self.up_factor, mode="nearest")
        z = self.conv2(F.silu(self.norm2(z)))

        # ensure residual matches input spatial size
        if z.shape[-2:] != x.shape[-2:]:
            z = F.interpolate(z, size=(h, w), mode="nearest")
        return z


class _Attn(nn.Module):
    """Lightweight multi-head spatial attention used inside the adaptor."""

    def __init__(self, ch: int, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.scale = (ch // heads) ** -0.5
        self.norm = _norm(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = qkv.unbind(dim=1)
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


# ---------------------------------------------------------------------
# Adapted U-Net: wraps a frozen U-Net and inserts an Adaptor at every
# Down/Up/Mid block. Forward signature is unchanged: (x_t, t) -> ε_hat.
# ---------------------------------------------------------------------
class AdaptedUNet(nn.Module):
    """Frozen U-Net θ + trainable adaptors ψ.

    Implements the residual rule  x_t^l = θ^l(x_t^{l-1}) + ψ^l(x_t^{l-1})
    by registering one Adaptor per Down / Up block (and one for the
    bottleneck) and adding its output to the corresponding block output.
    """

    def __init__(
        self,
        unet: UNet,
        c: int = 4,
        d: int = 8,
        num_heads: int = 4,
        zero_init: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.unet = unet

        if freeze_backbone:
            for p in self.unet.parameters():
                p.requires_grad_(False)

        # Build one adaptor per encoder block, decoder block, and the
        # bottleneck. Channel dims are inferred lazily on the first
        # forward pass via hooks; for simplicity we use the input
        # channel of each block.
        self.down_adaptors: nn.ModuleList = self._build_adaptors_for(
            unet.down_blocks, c, d, num_heads, zero_init
        )
        self.mid_adaptor = Adaptor(
            self._infer_channels(unet.mid), c, d, num_heads, zero_init
        )
        self.up_adaptors: nn.ModuleList = self._build_adaptors_for(
            unet.up_blocks, c, d, num_heads, zero_init
        )

    @staticmethod
    def _infer_channels(block) -> int:
        for layer in block.modules():
            if isinstance(layer, nn.Conv2d):
                return layer.out_channels
        return 128

    @classmethod
    def _build_adaptors_for(
        cls, blocks: nn.ModuleList, c, d, num_heads, zero_init
    ) -> nn.ModuleList:
        adapters: List[nn.Module] = []
        for block in blocks:
            adapters.append(
                Adaptor(cls._infer_channels(block), c, d, num_heads, zero_init)
            )
        return nn.ModuleList(adapters)

    # ---------------------------------------------------------------
    def trainable_parameters(self):
        """Yield only adaptor parameters (paper §4.3 freezes θ)."""
        for p in self.down_adaptors.parameters():
            yield p
        for p in self.mid_adaptor.parameters():
            yield p
        for p in self.up_adaptors.parameters():
            yield p

    # ---------------------------------------------------------------
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """ε_{θ,ψ}(x_t, t).  Mirrors UNet.forward but adds ψ residuals."""
        un = self.unet
        t_emb = un.t_embed(t)
        h = un.in_conv(x)
        skips = [h]

        for block, adp in zip(un.down_blocks, self.down_adaptors):
            h_in = h
            for layer in block:
                from .unet import ResBlock as _RB

                h = layer(h, t_emb) if isinstance(layer, _RB) else layer(h)
            h = h + adp(h_in)
            skips.append(h)

        h_in = h
        for layer in un.mid:
            from .unet import ResBlock as _RB

            h = layer(h, t_emb) if isinstance(layer, _RB) else layer(h)
        h = h + self.mid_adaptor(h_in)

        for block, adp in zip(un.up_blocks, self.up_adaptors):
            from .unet import ResBlock as _RB

            first = block[0]
            h_in = h
            if isinstance(first, _RB):
                h_cat = torch.cat([h, skips.pop()], dim=1)
                h = first(h_cat, t_emb)
                for layer in block[1:]:
                    h = layer(h, t_emb) if isinstance(layer, _RB) else layer(h)
            else:
                for layer in block:
                    h = layer(h)
            h = h + adp(h_in)

        h = F.silu(un.out_norm(h))
        return un.out_conv(h)
