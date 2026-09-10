"""CLIP vision-tower loader using open_clip.

The paper uses OpenAI ViT-L/14 @ 224 as the canonical encoder
(addendum: "OpenAI CLIP ViT-L/14@224 vision encoder rather than the default
ViT-L/14@336"). We use open_clip because the addendum explicitly states the
LLaVA code was modified to consume the open_clip implementation rather than
the HuggingFace one.

Two encoders are instantiated:
  * `original_visual` — frozen reference phi_Org (eval mode, no grad)
  * `finetune_visual` — trainable copy phi_FT (train mode)

Both start from identical OpenAI weights so phi_FT(x) = phi_Org(x) at step 0.
"""

from __future__ import annotations

import copy
from typing import Tuple

import torch
import torch.nn as nn


def _try_import_open_clip():
    try:
        import open_clip  # type: ignore

        return open_clip
    except ImportError as exc:
        raise ImportError(
            "open_clip is required. Install via `pip install open_clip_torch`."
        ) from exc


class CLIPVisionWrapper(nn.Module):
    """Wraps an open_clip vision tower, returning the *unnormalized* class-token
    embedding (Sec. B.1 of the paper).

    The forward expects images already in the standard CLIP-normalized form
    (mean/std of OpenAI weights). Adversarial perturbations are added in pixel
    space [0, 1] BEFORE this normalization is applied (per FARE convention and
    the addendum line: "computation of l_infinity ball around non-normalized
    inputs").
    """

    def __init__(
        self,
        visual: nn.Module,
        image_mean: Tuple[float, float, float],
        image_std: Tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.visual = visual
        # Register normalization tensors as buffers so they move with .to(device)
        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std).view(1, 3, 1, 1),
            persistent=False,
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CLIP normalization. `x` is in [0, 1]."""
        return (x - self.image_mean) / self.image_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: pixel-space tensor in [0, 1]. Returns class-token embedding."""
        x_norm = self.normalize(x)
        # open_clip's `visual` returns the projected class-token by default.
        return self.visual(x_norm)


def load_clip_vision(
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
    device: str | torch.device = "cuda",
) -> Tuple[CLIPVisionWrapper, CLIPVisionWrapper, int]:
    """Load matched (frozen, trainable) CLIP vision towers.

    Returns
    -------
    original : frozen reference, eval-mode, requires_grad=False
    finetune : trainable copy, train-mode
    image_resolution : input resolution expected by the model (224 for ViT-L/14)
    """
    open_clip = _try_import_open_clip()

    full_model, _, _ = open_clip.create_model_and_transforms(
        model_name=model_name, pretrained=pretrained
    )
    visual = full_model.visual

    # Mean/std comes from open_clip's preprocessing constants.
    image_mean = getattr(visual, "image_mean", (0.48145466, 0.4578275, 0.40821073))
    image_std = getattr(visual, "image_std", (0.26862954, 0.26130258, 0.27577711))
    resolution = int(getattr(visual, "image_size", 224))
    if isinstance(resolution, (list, tuple)):
        resolution = int(resolution[0])

    finetune = CLIPVisionWrapper(visual, image_mean, image_std)
    original = CLIPVisionWrapper(copy.deepcopy(visual), image_mean, image_std)

    # Freeze the reference encoder.
    for p in original.parameters():
        p.requires_grad_(False)
    original.eval()

    finetune.to(device)
    original.to(device)

    return original, finetune, resolution
