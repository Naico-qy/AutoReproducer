"""Zero-shot CLIP classifier built from text prompts.

Standard CLIP zero-shot recipe (Radford et al., 2021):
  1. For each class c, encode a small set of prompt templates
     "a photo of a {c}", "a picture of a {c}", ...
  2. Average per-class embeddings, L2-normalize.
  3. Classification logits = visual_features @ text_features^T * scale.

For evaluation we wrap this into a torch.nn.Module so attacks can use it
just like any other classifier.
"""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# OpenAI CLIP-benchmark default templates (subset). Used by Sec. 4.3 evals.
DEFAULT_TEMPLATES: List[str] = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a picture of a {}.",
    "an image of a {}.",
    "a close-up photo of a {}.",
    "a low resolution photo of a {}.",
    "a bright photo of a {}.",
    "a dark photo of a {}.",
    "a cropped photo of a {}.",
    "a blurry photo of a {}.",
]


class ZeroShotClassifier(nn.Module):
    """Wraps a (vision_encoder, text_classifier_weights, logit_scale) triple.

    The vision_encoder is expected to map pixel-space [0, 1] inputs to
    un-normalized embeddings. We L2-normalize internally before computing
    logits, matching the OpenAI CLIP convention.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_weights: torch.Tensor,  # (D, num_classes), L2-normalized
        logit_scale: float = 100.0,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.register_buffer("text_weights", text_weights, persistent=False)
        self.logit_scale = logit_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.vision_encoder(x)
        feats = F.normalize(feats, dim=-1)
        return self.logit_scale * feats @ self.text_weights


@torch.no_grad()
def build_zeroshot_classifier(
    open_clip_text_encoder,
    open_clip_tokenizer,
    classnames: Iterable[str],
    templates: Iterable[str] = DEFAULT_TEMPLATES,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Construct the (D, K) text-classifier weight matrix.

    Parameters
    ----------
    open_clip_text_encoder : a callable taking tokenized prompts -> features.
                             For an open_clip CLIP model, this is the bound
                             method `model.encode_text`.
    open_clip_tokenizer    : open_clip.get_tokenizer(model_name).
    classnames             : iterable of class strings.
    templates              : iterable of prompt template strings (with `{}`).
    """
    text_weights = []
    for cname in classnames:
        prompts = [t.format(cname) for t in templates]
        tokens = open_clip_tokenizer(prompts).to(device)
        emb = open_clip_text_encoder(tokens)
        emb = F.normalize(emb, dim=-1)
        # Mean across templates, then re-normalize.
        cls_emb = emb.mean(dim=0)
        cls_emb = cls_emb / cls_emb.norm()
        text_weights.append(cls_emb)
    return torch.stack(text_weights, dim=1)  # (D, K)
