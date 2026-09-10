"""U-Net noise predictor ε_θ(x_t, t).

Implements the ADM-style U-Net (Dhariwal & Nichol, NeurIPS 2021), which is
the backbone used in the paper for the DDPM framework (paper §5.2:
"we employ a pre-trained DDPM similar to DDPM-PA"). The same module is
also used as the latent denoiser inside the LDM framework (Rombach et al.
2022); for LDM the input is a 4-channel z-latent at 64x64 resolution.

The forward signature matches paper Eq. (1):  eps_theta(x_t, t) -> eps_hat
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------
# Sinusoidal time embedding (Vaswani / Ho et al.)
# ---------------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


def _norm(c: int) -> nn.Module:
    return nn.GroupNorm(num_groups=min(32, c), num_channels=c, eps=1e-6)


class ResBlock(nn.Module):
    """Residual block with FiLM-style time conditioning."""

    def __init__(
        self, in_ch: int, out_ch: int, t_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = _norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = _norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention over the spatial map."""

    def __init__(self, ch: int, num_heads: int = 4) -> None:
        super().__init__()
        assert ch % num_heads == 0
        self.heads = num_heads
        self.norm = _norm(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = qkv.unbind(dim=1)
        scale = (c // self.heads) ** -0.5
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """ADM-style U-Net for ε prediction.

    Args mirror the YAML config (configs/default.yaml :: unet)."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult=(1, 1, 2, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions=(16, 8),
        image_size: int = 256,
        dropout: float = 0.0,
        num_heads: int = 4,
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

        # Encoder
        self.down_blocks = nn.ModuleList()
        ch_in = base_channels
        chs = [ch_in]
        cur_res = image_size
        for i, mult in enumerate(channel_mult):
            ch_out = base_channels * mult
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(ch_in, ch_out, t_dim, dropout)])
                if cur_res in attention_resolutions:
                    block.append(AttentionBlock(ch_out, num_heads))
                self.down_blocks.append(block)
                ch_in = ch_out
                chs.append(ch_in)
            if i != len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(ch_in)]))
                chs.append(ch_in)
                cur_res //= 2

        # Bottleneck
        self.mid = nn.ModuleList(
            [
                ResBlock(ch_in, ch_in, t_dim, dropout),
                AttentionBlock(ch_in, num_heads),
                ResBlock(ch_in, ch_in, t_dim, dropout),
            ]
        )

        # Decoder
        self.up_blocks = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mult)):
            ch_out = base_channels * mult
            for _ in range(num_res_blocks + 1):
                skip = chs.pop()
                block = nn.ModuleList([ResBlock(ch_in + skip, ch_out, t_dim, dropout)])
                if cur_res in attention_resolutions:
                    block.append(AttentionBlock(ch_out, num_heads))
                self.up_blocks.append(block)
                ch_in = ch_out
            if i != len(channel_mult) - 1:
                self.up_blocks.append(nn.ModuleList([Upsample(ch_in)]))
                cur_res *= 2

        self.out_norm = _norm(ch_in)
        self.out_conv = nn.Conv2d(ch_in, out_channels, 3, padding=1)

    # -----------------------------------------------------------------
    # Forward: ε_θ(x_t, t)
    # -----------------------------------------------------------------
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_emb = self.t_embed(t)
        h = self.in_conv(x)
        skips = [h]
        for block in self.down_blocks:
            for layer in block:
                h = layer(h, t_emb) if isinstance(layer, ResBlock) else layer(h)
            skips.append(h)
        for layer in self.mid:
            h = layer(h, t_emb) if isinstance(layer, ResBlock) else layer(h)
        for block in self.up_blocks:
            first = block[0]
            if isinstance(first, ResBlock):
                h = torch.cat([h, skips.pop()], dim=1)
                h = first(h, t_emb)
                for layer in block[1:]:
                    h = layer(h, t_emb) if isinstance(layer, ResBlock) else layer(h)
            else:
                for layer in block:
                    h = layer(h)
        h = F.silu(self.out_norm(h))
        return self.out_conv(h)
