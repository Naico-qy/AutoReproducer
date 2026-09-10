"""Functional adapter implementations for SEMA (Sec. 3.3, Eq. 1).

Three variants are supported, all sharing the same forward signature so that
they can be plugged into a `ModularAdapterBlock` interchangeably:

* Adapter (Chen et al., 2022 -- AdaptFormer)
    f_phi(x) = ReLU(x . W_down) . W_up      (Eq. 1)
    Default bottleneck r = 48 (per addendum).
    The reference implementation that ADAM uses is taken from:
    https://github.com/ShoufaChen/AdaptFormer/blob/main/models/adapter.py

* LoRA (Hu et al., 2021)
    f_phi(x) = (x . A) . B   (no non-linearity, low-rank)

* Convpass (Jie & Deng, 2022)
    Convolutional bypass adapter -- treats the token sequence as a 2D feature
    map and applies a small 3x3 conv between two linear projections.
    Reference: https://github.com/JieShibo/PETL-ViT/blob/main/convpass/vtab/convpass.py

CrossRef-verified reference (used as the default adapter in SEMA):
    Chen, S., Ge, C., Tong, Z., Wang, J., Song, Y., Wang, J., Luo, P.
    AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition.
    NeurIPS 2022.
    URL: papers.nips.cc/paper_files/paper/2022/hash/69e2f49ab0837b71b0e0cb7c555990f8
"""

from __future__ import annotations

import math

import torch
from torch import nn


class FunctionalAdapter(nn.Module):
    """AdaptFormer-style bottleneck adapter (Eq. 1).

    f_phi(x) = ReLU(x . W_down) . W_up
    Default bottleneck r = 48 (addendum). Down-/up-projections are linear with
    no bias by default, matching the reference AdaptFormer implementation.
    """

    def __init__(self, dim: int, bottleneck: int = 48, scale: float = 1.0) -> None:
        super().__init__()
        self.dim = dim
        self.bottleneck = bottleneck
        self.scale = scale
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, dim, bias=False)
        self.act = nn.ReLU(inplace=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Kaiming uniform on down; zero-init on up, mirroring AdaptFormer.
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.up(self.act(self.down(x)))


class LoRAAdapter(nn.Module):
    """LoRA-style low-rank adapter (Hu et al., 2021).

    f_phi(x) = alpha/r * (x . A . B)
    """

    def __init__(self, dim: int, bottleneck: int = 48, alpha: float = 1.0) -> None:
        super().__init__()
        self.dim = dim
        self.bottleneck = bottleneck
        self.scale = alpha / max(bottleneck, 1)
        self.A = nn.Linear(dim, bottleneck, bias=False)
        self.B = nn.Linear(bottleneck, dim, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.B(self.A(x))


class ConvPassAdapter(nn.Module):
    """Convpass adapter (Jie & Deng, 2022).

    Treats the (B, N, D) token tensor as a (B, D, H, W) feature map (skipping
    the CLS token), applies a 3x3 conv between two 1x1 linear projections, and
    folds the output back into the (B, N, D) shape. The CLS token is passed
    through unchanged.
    """

    def __init__(self, dim: int, bottleneck: int = 48, n_tokens: int = 196) -> None:
        super().__init__()
        self.dim = dim
        self.bottleneck = bottleneck
        # Default ViT-B/16 with 224x224 input -> 14x14=196 patch tokens.
        side = int(math.sqrt(n_tokens))
        if side * side != n_tokens:
            side = int(math.floor(math.sqrt(n_tokens)))
        self.h = self.w = side
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.conv = nn.Conv2d(
            bottleneck, bottleneck, kernel_size=3, padding=1, bias=False
        )
        self.up = nn.Linear(bottleneck, dim, bias=False)
        self.act = nn.GELU()
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D); split CLS and patch tokens.
        cls, patches = x[:, :1, :], x[:, 1:, :]
        z = self.down(patches)  # (B, N-1, r)
        b, n, r = z.shape
        h = w = self.h
        if n != h * w:
            # Fallback: pad/truncate to nearest square.
            h = w = int(math.floor(math.sqrt(n)))
            n = h * w
            z = z[:, :n, :]
        z = z.transpose(1, 2).reshape(b, r, h, w)  # (B, r, H, W)
        z = self.act(self.conv(z))
        z = z.reshape(b, r, h * w).transpose(1, 2)  # (B, N, r)
        z = self.up(z)
        out = torch.zeros_like(x)
        out[:, 1:, :][:, : z.shape[1], :] = z
        return out


def build_adapter(kind: str, dim: int, bottleneck: int = 48) -> nn.Module:
    """Factory for the supported adapter variants.

    Args:
        kind: One of {"adapter", "lora", "convpass"}.
        dim: Input/output feature dimension.
        bottleneck: Bottleneck dimension r (default 48 per addendum).
    """
    kind = kind.lower()
    if kind in {"adapter", "adaptformer"}:
        return FunctionalAdapter(dim=dim, bottleneck=bottleneck)
    if kind == "lora":
        return LoRAAdapter(dim=dim, bottleneck=bottleneck)
    if kind == "convpass":
        return ConvPassAdapter(dim=dim, bottleneck=bottleneck)
    raise ValueError(f"Unknown adapter kind: {kind}")
