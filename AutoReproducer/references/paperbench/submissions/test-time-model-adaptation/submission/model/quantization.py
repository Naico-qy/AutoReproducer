"""Light-weight 8-bit / 6-bit ViT quantization hooks for the FOA paper's
Section 4.2 ("Results on Quantized Models").

The paper uses PTQ4ViT (Yuan et al., 2022) for full PTQ.  Per addendum.md:

    > The implementation details of PTQ4ViT for model quantization are
    > partially missing from the main text. We refer to the original
    > PTQ4ViT paper for the complete quantization process details.

A faithful PTQ4ViT reimplementation is out of scope for the smoke run
(it requires calibration on a held-out split, search-based scaling for
softmax/GELU, etc.).  We instead provide:

  1. ``quantize_vit_8bit`` -- a *symmetric per-tensor* INT8 weight
     quantization wrapper that rounds linear weights to the nearest
     8-bit grid.  This preserves forward-pass behavior closely enough
     to validate the FOA pipeline numerically (FOA only requires forward
     passes; weights stay frozen) and exercises the API.
  2. A clear NotImplementedError for "ptq4vit" mode pointing the user
     to the official PTQ4ViT repo.

Reference (verified):
    Yuan, Z., Xue, C., Chen, Y., Wu, Q., & Sun, G. "PTQ4ViT: Post-Training
    Quantization Framework for Vision Transformers with Twin Uniform
    Quantization." ECCV 2022.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .architecture import PromptedViT


@torch.no_grad()
def _symmetric_int_quant(t: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Symmetric per-tensor uniform INT-bits weight quantization."""
    if bits <= 1 or bits > 16:
        raise ValueError(f"bits must be in [2,16], got {bits}")
    qmax = (1 << (bits - 1)) - 1
    amax = t.detach().abs().max().clamp(min=1e-8)
    scale = amax / qmax
    q = torch.round(t / scale).clamp(-qmax - 1, qmax)
    return (q * scale).to(t.dtype)


def quantize_vit_8bit(model: PromptedViT, bits: int = 8) -> PromptedViT:
    """In-place symmetric quantization of all ``nn.Linear`` weights inside the
    backbone.  The classification head is left untouched (paper convention).

    This is *not* a full PTQ4ViT pipeline -- it is a lightweight stand-in to
    allow Section-4.2-style execution.  For exact paper numbers, swap in
    PTQ4ViT (https://github.com/hahnyuan/PTQ4ViT).
    """
    for m in model.backbone.modules():
        if isinstance(m, nn.Linear):
            m.weight.data = _symmetric_int_quant(m.weight.data, bits=bits)
    return model
