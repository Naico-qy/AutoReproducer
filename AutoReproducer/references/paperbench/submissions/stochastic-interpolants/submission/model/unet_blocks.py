"""Lower-level U-Net building blocks.

These mirror the *lucidrains/denoising-diffusion-pytorch* implementation
that the paper's Appendix B explicitly references for its velocity
network. We re-implement them here so the codebase has zero proprietary
dependencies but stays bit-compatible with that architecture's choices.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Time-conditioning embeddings ----------------------------------------------
# ---------------------------------------------------------------------------


class SinusoidalPosEmb(nn.Module):
    """Original Transformer sinusoidal embedding for the time scalar."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(-freqs * torch.arange(half, device=t.device, dtype=t.dtype))
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class LearnedSinusoidalPosEmb(nn.Module):
    """Learned-Fourier time embedding (Appendix B → True, dim 32)."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.weights = nn.Parameter(torch.randn(dim // 2))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = t[:, None] * self.weights[None, :] * 2 * math.pi
        fouriered = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return torch.cat([t[:, None], fouriered], dim=-1)


# ---------------------------------------------------------------------------
# ResNet block --------------------------------------------------------------
# ---------------------------------------------------------------------------


class Block(nn.Module):
    """Conv-GroupNorm-SiLU; `groups` controlled from config (=8 in paper)."""

    def __init__(self, dim_in: int, dim_out: int, groups: int = 8):
        super().__init__()
        # If dim_out is small, fall back to the largest legal #groups.
        groups = math.gcd(groups, dim_out)
        groups = max(1, groups)
        self.proj = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift: Optional[tuple] = None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        time_emb_dim: Optional[int] = None,
        class_emb_dim: Optional[int] = None,
        groups: int = 8,
    ):
        super().__init__()
        cond_dim = 0
        if time_emb_dim is not None:
            cond_dim += time_emb_dim
        if class_emb_dim is not None:
            cond_dim += class_emb_dim
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim_out * 2))
            if cond_dim
            else None
        )
        self.block1 = Block(dim_in, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = (
            nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        )

    def forward(self, x, time_emb=None, class_emb=None):
        scale_shift = None
        if self.mlp is not None:
            cond_parts = []
            if time_emb is not None:
                cond_parts.append(time_emb)
            if class_emb is not None:
                cond_parts.append(class_emb)
            cond = torch.cat(cond_parts, dim=-1)
            cond = self.mlp(cond)
            cond = rearrange(cond, "b c -> b c 1 1")
            scale, shift = cond.chunk(2, dim=1)
            scale_shift = (scale, shift)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


# ---------------------------------------------------------------------------
# Multi-head self-attention block (Appendix B: dim_head=64, heads=4) --------
# ---------------------------------------------------------------------------


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 64):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads),
            qkv,
        )
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    """Standard quadratic attention used at the bottleneck."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 64):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads),
            qkv,
        )
        q = q * self.scale
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) c -> b (h c) x y", x=h, y=w)
        return self.to_out(out)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


# ---------------------------------------------------------------------------
# Up / Down samplers (lucidrains style: pixel-shuffle / -unshuffle) ---------
# ---------------------------------------------------------------------------


def Downsample(dim_in: int, dim_out: Optional[int] = None) -> nn.Module:
    dim_out = dim_out or dim_in
    return nn.Sequential(
        nn.PixelUnshuffle(2),
        nn.Conv2d(dim_in * 4, dim_out, 1),
    )


def Upsample(dim_in: int, dim_out: Optional[int] = None) -> nn.Module:
    dim_out = dim_out or dim_in
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim_in, dim_out, 3, padding=1),
    )
