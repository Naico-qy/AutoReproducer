"""Prompted Vision Transformer for FOA test-time adaptation.

Implements the architecture described in Section 3.1 of:

    Niu, S., Miao, C., Chen, G., Wu, P., & Zhao, P. (2024).
    Test-Time Model Adaptation with Only Forward Passes. ICML 2024.

Key design choices grounded in the paper / addendum:

- ViT-Base backbone from `timm` (`vit_base_patch16_224`), trained on ImageNet-1K
  (Section 4 "Datasets and Models", paragraph: "the model weights are obtained
  from the timm repository (Wightman, 2019)").
- N_p = 3 learnable prompt embeddings, dim = 768 (Section 4 "Implementation
  Details").
- Prompts are inserted *between* the [CLS] token and the patch embeddings, in
  the order [CLS, prompts, patches] -- this is explicitly stated in
  addendum.md: "The arrangement of input sequence elements is [CLS token,
  learnable prompts, patch embeddings] in that specific order."
- The forward pass exposes per-layer CLS features {e_i^0}_{i=1..N} for the
  fitness function in Eqn. (5).
- All backbone weights are frozen (paper: "while keeping all other model
  parameters frozen.").

References (verified):
    Wang, Dequan; Shelhamer, Evan; Liu, Shaoteng; Olshausen, Bruno;
    Darrell, Trevor. "Tent: Fully Test-Time Adaptation by Entropy
    Minimization." ICLR 2021. (verified in session via paper_search)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _maybe_import_timm():
    try:
        import timm  # type: ignore

        return timm
    except Exception:  # pragma: no cover
        return None


class PromptedViT(nn.Module):
    """A wrapper around a frozen timm ViT that splices learnable prompts.

    The wrapped model implements the standard ViT forward (Eqn. (1)-(2) of the
    paper):

        E_i = L_i(E_{i-1}),   i = 1, ..., N
        y_hat = Head(e_N^0)

    but inserts ``num_prompts`` extra learnable embeddings p between the [CLS]
    token and the patch embeddings.  We keep ``forward_with_prompt`` differen-
    tiable so that we can also fit the SGD-prompt baseline used in Table 9 of
    the paper, but for FOA itself ``no_grad`` is used at the call-site.

    Parameters
    ----------
    model_name : str
        ``timm`` model identifier. Default ``vit_base_patch16_224`` matches
        the paper's "ViT-Base" model.
    num_prompts : int
        Number of prompt embeddings ``N_p``.  Paper default = 3.
    pretrained : bool
        Whether to load ImageNet-1K pretrained weights.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_prompts: int = 3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        timm = _maybe_import_timm()
        if timm is None:
            raise RuntimeError("timm is required. Install via `pip install timm`.")
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        # Freeze all backbone parameters -- FOA does NOT update model weights.
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.embed_dim: int = self.backbone.embed_dim  # 768 for ViT-Base
        self.num_prompts = num_prompts
        # Default prompt initialization: small uniform (paper Section 4
        # "Implementation Details": initialize prompts with uniform init).
        bound = 1.0 / float(self.embed_dim) ** 0.5
        prompt = torch.empty(num_prompts, self.embed_dim).uniform_(-bound, bound)
        self.prompt = nn.Parameter(prompt, requires_grad=True)

        # Cache references to internal timm components for easy access.
        self.cls_token = self.backbone.cls_token  # (1,1,D)
        self.pos_embed = self.backbone.pos_embed  # (1, 1+num_patches, D)
        self.patch_embed = self.backbone.patch_embed
        self.blocks = self.backbone.blocks
        # timm uses `norm` as the final layer norm before head
        self.norm = self.backbone.norm
        self.head = self.backbone.head

    # ------------------------------------------------------------------
    # Prompt management
    # ------------------------------------------------------------------
    def get_prompt_dim(self) -> int:
        """Total dim of flattened prompt vector p in R^{d * N_p}."""
        return self.num_prompts * self.embed_dim

    def set_prompt(self, p_flat: torch.Tensor) -> None:
        """Set prompt from a flat vector (used by CMA-ES sampling)."""
        with torch.no_grad():
            self.prompt.copy_(
                p_flat.view(self.num_prompts, self.embed_dim).to(self.prompt)
            )

    def reset_prompt(self) -> None:
        bound = 1.0 / float(self.embed_dim) ** 0.5
        with torch.no_grad():
            self.prompt.uniform_(-bound, bound)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _build_input_tokens(
        self, x: torch.Tensor, prompt_override: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Construct the input sequence: [CLS, prompts, patches] (per addendum)."""
        B = x.shape[0]
        patches = self.patch_embed(x)  # (B, num_patches, D)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        if prompt_override is None:
            prm = self.prompt  # (N_p, D)
        else:
            prm = prompt_override
        prm = prm.unsqueeze(0).expand(B, -1, -1)  # (B, N_p, D)
        # Order: [CLS, prompts, patches]  (addendum.md, line: "The arrangement
        # of input sequence elements is [CLS token, learnable prompts, patch
        # embeddings] in that specific order.")
        tokens = torch.cat([cls, prm, patches], dim=1)
        # Add positional embeddings.  timm pos_embed has shape (1, 1+P, D)
        # covering CLS + patches.  We extend it with zeros for the prompt
        # positions so that prompts learn their own absolute representation.
        pe = self.pos_embed  # (1, 1+P, D)
        cls_pe = pe[:, :1, :]
        patch_pe = pe[:, 1:, :]
        prompt_pe = torch.zeros(
            1, self.num_prompts, self.embed_dim, device=pe.device, dtype=pe.dtype
        )
        full_pe = torch.cat([cls_pe, prompt_pe, patch_pe], dim=1)
        tokens = tokens + full_pe
        if hasattr(self.backbone, "pos_drop"):
            tokens = self.backbone.pos_drop(tokens)
        return tokens

    def forward_features(
        self,
        x: torch.Tensor,
        prompt_override: Optional[torch.Tensor] = None,
        return_all_cls: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the ViT forward, returning final CLS feature and list of
        per-layer CLS features (for the fitness function).

        The CLS token always sits at index 0 regardless of how many prompts are
        inserted, so we extract feats[:, 0] at every depth.
        """
        tokens = self._build_input_tokens(x, prompt_override)
        cls_feats: List[torch.Tensor] = []
        h = tokens
        for blk in self.blocks:
            h = blk(h)
            if return_all_cls:
                cls_feats.append(h[:, 0])  # (B, D)
        h = self.norm(h)
        cls_final = h[:, 0]  # (B, D)
        # Replace last layer feature with normed version for consistency.
        if return_all_cls and len(cls_feats) > 0:
            cls_feats[-1] = cls_final
        return cls_final, cls_feats

    def classify(self, cls_feature: torch.Tensor) -> torch.Tensor:
        """Apply the (frozen) classification head."""
        return self.head(cls_feature)

    def forward(
        self,
        x: torch.Tensor,
        prompt_override: Optional[torch.Tensor] = None,
        return_all_cls: bool = False,
        cls_offset: Optional[torch.Tensor] = None,
    ):
        """Standard forward pass.

        Parameters
        ----------
        x : (B, 3, H, W) image tensor
        prompt_override : optional flat or (N_p, D) tensor, used by CMA-ES
        return_all_cls : if True, return per-layer CLS features for fitness
        cls_offset : optional (D,) or (B,D) offset to add to e_N^0 before the
            head, used to implement the back-to-source activation shifting
            (Eqn. (7)).
        """
        if prompt_override is not None and prompt_override.dim() == 1:
            prompt_override = prompt_override.view(self.num_prompts, self.embed_dim)
        cls_final, cls_feats = self.forward_features(
            x, prompt_override=prompt_override, return_all_cls=return_all_cls
        )
        if cls_offset is not None:
            if cls_offset.dim() == 1:
                cls_final = cls_final + cls_offset.unsqueeze(0)
            else:
                cls_final = cls_final + cls_offset
        logits = self.classify(cls_final)
        if return_all_cls:
            return logits, cls_final, cls_feats
        return logits


def build_vit_base(num_prompts: int = 3, pretrained: bool = True) -> PromptedViT:
    """Convenience builder for paper-default ViT-Base/16 with 3 prompts."""
    return PromptedViT(
        model_name="vit_base_patch16_224",
        num_prompts=num_prompts,
        pretrained=pretrained,
    )
