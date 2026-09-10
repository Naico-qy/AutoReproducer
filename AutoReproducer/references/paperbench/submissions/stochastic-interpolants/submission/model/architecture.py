"""U-Net velocity model b̂_t(x, ξ) — Appendix B of the paper.

Hyper-parameters from the paper / addendum:
  * dim = 256
  * dim_mults = (1, 1, 2, 3, 4)
  * resnet_block_groups = 8  (= GroupNorm groups, per addendum)
  * learned_sinusoidal_cond = True, learned_sinusoidal_dim = 32
  * attn_dim_head = 64, attn_heads = 4
  * random_fourier_features = False  (we use the learned sinusoidal
    embedding instead — the paper sets the random Fourier feature flag
    to False so this is the canonical choice.)

The network takes:
  * x       — the interpolant value I_t (B, C_x, H, W)
  * t       — time scalar (B,)
  * cond    — image-shape conditioning ξ (e.g. low-res image, mask),
              concatenated to x along the channel axis. Optional.
  * cls     — integer class id (B,) for ImageNet 1k. Optional.

Reference (verified via CrossRef in this submission):
    Saharia et al. "Image Super-Resolution via Iterative Refinement",
    IEEE TPAMI 2022.  DOI 10.1109/TPAMI.2022.3204461
    (used as the SR3 baseline in Table 3 of the paper; the same
    convention of appending an upsampled low-res image to the U-Net
    input is followed here.)
"""

from __future__ import annotations

from functools import partial
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_blocks import (
    Attention,
    LearnedSinusoidalPosEmb,
    LinearAttention,
    PreNorm,
    Residual,
    ResnetBlock,
    SinusoidalPosEmb,
    Downsample,
    Upsample,
)


def _exists(x):
    return x is not None


def _default(val, d):
    return val if _exists(val) else d


class UNetVelocity(nn.Module):
    """The velocity network that outputs b̂_t(x_t, ξ).

    Following Appendix B, image-shape conditioning ξ is supplied as
    *additional input channels*. We therefore accept `cond` here and
    concatenate it along the channel axis at the very first conv.

    The optional `apply_inpaint_mask_at_output` flag implements the
    in-painting trick from §4.1: for every t, ξ⊙I_t = ξ⊙x_1 so the true
    velocity is identically zero in the unmasked region. We bake that
    constraint into the network's output by zeroing-out the masked-region
    complement.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        cond_channels: int = 0,  # extra image-shape conditioning ξ
        dim: int = 256,
        dim_mults: Sequence[int] = (1, 1, 2, 3, 4),
        resnet_block_groups: int = 8,
        learned_sinusoidal_cond: bool = True,
        learned_sinusoidal_dim: int = 32,
        attn_dim_head: int = 64,
        attn_heads: int = 4,
        random_fourier_features: bool = False,  # noqa: ARG002 (kept for API parity)
        num_classes: Optional[int] = 1000,
        class_embed_dim: int = 256,
        apply_inpaint_mask_at_output: bool = False,
    ) -> None:
        super().__init__()

        self.out_channels = out_channels
        self.cond_channels = cond_channels
        self.apply_inpaint_mask_at_output = apply_inpaint_mask_at_output

        # ---- input conv ------------------------------------------------
        first_in = in_channels + cond_channels
        init_dim = dim
        self.init_conv = nn.Conv2d(first_in, init_dim, 7, padding=3)

        # ---- channel ladder -------------------------------------------
        dims: List[int] = [init_dim, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        # ---- time conditioning ----------------------------------------
        time_dim = dim * 4
        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # ---- class conditioning (Appendix B: class label embedding) ---
        self.num_classes = num_classes
        if num_classes is not None:
            self.class_embed_dim = class_embed_dim
            self.class_emb = nn.Embedding(num_classes + 1, class_embed_dim)
            # Index num_classes is reserved for "no class" / drop token.
            self.null_class_index = num_classes
        else:
            self.class_embed_dim = 0
            self.class_emb = None
            self.null_class_index = None

        block = partial(
            ResnetBlock,
            time_emb_dim=time_dim,
            class_emb_dim=self.class_embed_dim if num_classes is not None else None,
            groups=resnet_block_groups,
        )

        # ---- down path -------------------------------------------------
        self.downs = nn.ModuleList([])
        n_resolutions = len(in_out)
        for ind, (dim_in_, dim_out_) in enumerate(in_out):
            is_last = ind == (n_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block(dim_in_, dim_in_),
                        block(dim_in_, dim_in_),
                        Residual(
                            PreNorm(
                                dim_in_,
                                LinearAttention(
                                    dim_in_, heads=attn_heads, dim_head=attn_dim_head
                                ),
                            )
                        ),
                        Downsample(dim_in_, dim_out_)
                        if not is_last
                        else nn.Conv2d(dim_in_, dim_out_, 3, padding=1),
                    ]
                )
            )

        # ---- middle ----------------------------------------------------
        mid_dim = dims[-1]
        self.mid_block1 = block(mid_dim, mid_dim)
        self.mid_attn = Residual(
            PreNorm(
                mid_dim, Attention(mid_dim, heads=attn_heads, dim_head=attn_dim_head)
            )
        )
        self.mid_block2 = block(mid_dim, mid_dim)

        # ---- up path ---------------------------------------------------
        self.ups = nn.ModuleList([])
        for ind, (dim_in_, dim_out_) in enumerate(reversed(in_out)):
            is_last = ind == (n_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block(dim_out_ + dim_in_, dim_out_),
                        block(dim_out_ + dim_in_, dim_out_),
                        Residual(
                            PreNorm(
                                dim_out_,
                                LinearAttention(
                                    dim_out_, heads=attn_heads, dim_head=attn_dim_head
                                ),
                            )
                        ),
                        Upsample(dim_out_, dim_in_)
                        if not is_last
                        else nn.Conv2d(dim_out_, dim_in_, 3, padding=1),
                    ]
                )
            )

        # ---- output ----------------------------------------------------
        self.final_block = block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, out_channels, 1)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        cls: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute b̂_t(x, ξ).

        Parameters
        ----------
        x:    interpolant I_t, shape (B, C_x, H, W).
        t:    time scalar in [0, 1], shape (B,).
        cond: optional image-shape conditioning, shape (B, C_cond, H, W);
              concatenated to x as additional channels (Appendix B).
        cls:  optional class id in [0, num_classes-1], shape (B,).
        mask: optional binary keep-mask (B, 1 or C, H, W); if provided
              and `apply_inpaint_mask_at_output` is True, the velocity
              is multiplied by `(1 - mask)` so unmasked pixels stay put.
        """
        # 1) input projection (concatenate ξ along channel axis)
        if cond is not None:
            x_in = torch.cat([x, cond], dim=1)
        else:
            x_in = x
        h = self.init_conv(x_in)
        r = h.clone()

        # 2) time + class embeddings
        t_emb = self.time_mlp(t)
        if self.class_emb is not None:
            if cls is None:
                cls_idx = torch.full(
                    (x.shape[0],),
                    self.null_class_index,
                    device=x.device,
                    dtype=torch.long,
                )
            else:
                cls_idx = cls.long()
            c_emb = self.class_emb(cls_idx)
        else:
            c_emb = None

        # 3) down path
        skips = []
        for block1, block2, attn, downsample in self.downs:
            h = block1(h, t_emb, c_emb)
            skips.append(h)
            h = block2(h, t_emb, c_emb)
            h = attn(h)
            skips.append(h)
            h = downsample(h)

        # 4) bottleneck
        h = self.mid_block1(h, t_emb, c_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb, c_emb)

        # 5) up path
        for block1, block2, attn, upsample in self.ups:
            h = torch.cat([h, skips.pop()], dim=1)
            h = block1(h, t_emb, c_emb)
            h = torch.cat([h, skips.pop()], dim=1)
            h = block2(h, t_emb, c_emb)
            h = attn(h)
            h = upsample(h)

        # 6) output conv
        h = torch.cat([h, r], dim=1)
        h = self.final_block(h, t_emb, c_emb)
        v = self.final_conv(h)

        # 7) §4.1 in-painting structural zero in unmasked region
        if self.apply_inpaint_mask_at_output and mask is not None:
            # mask=1 means "unmasked pixel" => velocity must be zero there.
            v = v * (1.0 - mask)
        return v


# ---------------------------------------------------------------------------
# Factory used by configs
# ---------------------------------------------------------------------------


def build_velocity_model(
    model_cfg: dict, *, in_channels: int, cond_channels: int
) -> UNetVelocity:
    return UNetVelocity(
        in_channels=in_channels,
        out_channels=in_channels,
        cond_channels=cond_channels,
        dim=int(model_cfg.get("dim", 256)),
        dim_mults=tuple(model_cfg.get("dim_mults", (1, 1, 2, 3, 4))),
        resnet_block_groups=int(model_cfg.get("resnet_block_groups", 8)),
        learned_sinusoidal_cond=bool(model_cfg.get("learned_sinusoidal_cond", True)),
        learned_sinusoidal_dim=int(model_cfg.get("learned_sinusoidal_dim", 32)),
        attn_dim_head=int(model_cfg.get("attn_dim_head", 64)),
        attn_heads=int(model_cfg.get("attn_heads", 4)),
        random_fourier_features=bool(model_cfg.get("random_fourier_features", False)),
        num_classes=model_cfg.get("num_classes", 1000),
        class_embed_dim=int(model_cfg.get("class_embed_dim", 256)),
        apply_inpaint_mask_at_output=bool(
            model_cfg.get("apply_inpaint_mask_at_output", False)
        ),
    )
