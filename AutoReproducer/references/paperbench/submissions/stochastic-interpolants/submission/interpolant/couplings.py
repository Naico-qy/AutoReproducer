"""Data-dependent couplings ρ(x_0, x_1) — Section 3.2 of the paper.

Three constructors are provided:

1. :class:`IndependentCoupling`  — the baseline `ρ(x_0,x_1) = ρ_0(x_0) ρ_1(x_1)`
   with `ρ_0 = N(0, Id)`. Used as the "Uncoupled Interpolant" baseline in
   Table 2 of the paper.

2. :class:`InpaintingCoupling`   — Section 4.1.
   `x_0 = ξ ⊙ x_1 + (1 − ξ) ⊙ ζ`, with `ζ ~ N(0, Id)` and ξ a binary
   tile mask drawn at runtime (64 tiles, p=0.3 per addendum).

3. :class:`SuperResolutionCoupling` — Section 4.2.
   `x_0 = U(D(x_1)) + σ ζ`, where `D` downsamples by a factor and `U`
   bicubically upsamples back to native resolution. The conditioning
   variable returned by the coupling is `ξ = U(D(x_1))`.

All couplings return a `(x_0, xi)` tuple — `xi` is fed to the velocity
network as additional channels per the paper / Appendix B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class Coupling:
    """Common interface — sample x_0 (and optionally ξ) given x_1."""

    def sample_x0(
        self, x1: torch.Tensor, *, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# (1) Independent coupling — baseline ----------------------------------------
# ---------------------------------------------------------------------------


@dataclass
class IndependentCoupling(Coupling):
    """ρ(x_0, x_1) = N(x_0; 0, Id) ρ_1(x_1) — the "Uncoupled" baseline."""

    def sample_x0(self, x1, *, generator=None):
        x0 = torch.randn_like(x1)
        return x0, None


# ---------------------------------------------------------------------------
# (2) In-painting coupling — Section 4.1 -------------------------------------
# ---------------------------------------------------------------------------


def _random_tile_mask(
    shape: Tuple[int, int, int, int],
    n_tiles: int = 64,
    p_mask: float = 0.3,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Random tile mask — Section 4.1.

    The image is split into `n_tiles` (= 64 by default) equally-sized
    tiles; each tile is set to *unmasked* (= 1) with probability
    `1 - p_mask` and *masked* (= 0) with probability `p_mask`.

    The output tensor has shape (B, 1, H, W) — broadcastable across all
    channels because the paper takes the same mask for every channel.
    """
    B, _, H, W = shape
    side = int(round(n_tiles**0.5))
    if side * side != n_tiles:
        raise ValueError(f"n_tiles must be a perfect square; got {n_tiles}")

    if H % side != 0 or W % side != 0:
        raise ValueError(
            f"image side ({H}, {W}) must be divisible by tile-grid side {side}"
        )
    tile_h, tile_w = H // side, W // side

    rand = torch.rand(
        (B, 1, side, side), device=device, dtype=dtype, generator=generator
    )
    keep = (rand >= p_mask).to(dtype)  # 1 where unmasked, 0 where masked
    keep = keep.repeat_interleave(tile_h, dim=2).repeat_interleave(tile_w, dim=3)
    return keep


@dataclass
class InpaintingCoupling(Coupling):
    """§4.1 — `x_0 = ξ ⊙ x_1 + (1 − ξ) ⊙ ζ`, ζ ~ N(0, Id).

    The conditioning variable returned is the binary mask ξ itself,
    matching the paper's claim that the network gets the mask but not
    the partial image (the partial image already lives inside x_t).
    """

    n_tiles: int = 64
    p_mask: float = 0.3

    def sample_x0(self, x1, *, generator=None):
        device = x1.device
        dtype = x1.dtype
        # Mask shape is broadcastable across channels — same value across C.
        keep = _random_tile_mask(
            (x1.shape[0], 1, x1.shape[-2], x1.shape[-1]),
            n_tiles=self.n_tiles,
            p_mask=self.p_mask,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        zeta = torch.randn_like(x1)
        x0 = keep * x1 + (1.0 - keep) * zeta
        # ξ in the paper's notation = the mask. Expand to match x1 channels
        # so it can be appended as image-shaped conditioning.
        xi = keep.expand_as(x1)
        return x0, xi


# ---------------------------------------------------------------------------
# (3) Super-resolution coupling — Section 4.2 --------------------------------
# ---------------------------------------------------------------------------


@dataclass
class SuperResolutionCoupling(Coupling):
    """§4.2 — `x_0 = U(D(x_1)) + σ ζ`, ζ ~ N(0, Id).

    `D` and `U` are average-pool / bilinear-upsample respectively. The
    conditioning variable ξ is the upsampled low-resolution image (which
    is what Appendix B says is appended to the velocity model's input).
    """

    low_res_size: int = 64  # 64 → 256 default (Table 3)
    sigma: float = 0.05
    interp: str = "bilinear"  # SR3 also uses bilinear/bicubic up.

    def sample_x0(self, x1, *, generator=None):
        H = x1.shape[-2]
        W = x1.shape[-1]
        # Down-sample (D) — area pool is the canonical anti-aliased downsampler.
        x_low = F.interpolate(
            x1, size=(self.low_res_size, self.low_res_size), mode="area"
        )
        # Up-sample (U) back to native resolution.
        x_up = F.interpolate(x_low, size=(H, W), mode=self.interp, align_corners=False)
        zeta = torch.randn_like(x1)
        x0 = x_up + self.sigma * zeta
        # ξ = U(D(x_1)) — image-shape conditioning.
        return x0, x_up


def make_coupling(cfg: dict) -> Coupling:
    """Factory used by configs."""
    kind = cfg.get("kind", "independent").lower()
    if kind == "independent":
        return IndependentCoupling()
    if kind == "inpainting":
        return InpaintingCoupling(
            n_tiles=int(cfg.get("inpaint_tiles", 64)),
            p_mask=float(cfg.get("inpaint_p_mask", 0.3)),
        )
    if kind == "superres":
        return SuperResolutionCoupling(
            low_res_size=int(cfg.get("low_res_size", 64)),
            sigma=float(cfg.get("sigma", 0.05)),
        )
    raise ValueError(f"Unknown coupling kind: {kind!r}")
