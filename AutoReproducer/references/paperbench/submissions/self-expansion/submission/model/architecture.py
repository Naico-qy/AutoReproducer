"""SEMA: Self-Expansion of Pre-trained Models with Modularised Adaptation.

Reference: Wang, H.; Lu, H.; Yao, L.; Gong, D.
"Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual
Learning". 2024.

This file implements the full SEMA model:
  * A frozen Vision Transformer backbone (ViT-B/16, IN-1K weights).
  * A `ModularAdapterBlock` attached to each expansion-enabled transformer
    layer (Sec. 3.3 / 3.4).
  * Hooks that feed pre-MLP features into the adapter mixture and add the
    mixture back to the post-MLP features (Eq. 3).
  * A nearest-prototype classifier head (matching SimpleCIL / ADAM convention,
    Sec. 4.2 / Appendix B).
  * The self-expansion strategy with z-score-based expansion signal and
    multi-layer (shallow -> deep) ordering (Sec. 3.6).

The implementation depends on `timm` for the ViT backbone, which mirrors the
reference codebases of ADAM, CODA-P and SEMA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .modular_block import ModularAdapterBlock


def _try_create_vit(
    name: str = "vit_base_patch16_224", pretrained: bool = True
) -> nn.Module:
    """Create a ViT backbone, falling back to a small placeholder ViT.

    We try `timm.create_model` first (canonical for ADAM/SEMA repos). If timm
    is not installed (e.g. unit-tests without network), a lightweight stand-in
    is returned that exposes the same interface used by SEMA: `patch_embed`,
    `pos_drop`, `blocks` (a sequential list of `nn.Module`s each having an
    `attn`, `norm1`, `mlp`, `norm2`), and `norm`/`head`.
    """
    try:  # noqa: SIM105
        import timm

        model = timm.create_model(name, pretrained=pretrained, num_classes=0)
        return model
    except Exception:  # pragma: no cover
        return _PlaceholderViT()


class _PlaceholderViT(nn.Module):
    """Tiny ViT-like network used when timm is unavailable.

    Exists only to keep `import` working in environments without timm; in any
    real run the timm-based ViT-B/16 is used.
    """

    def __init__(self, dim: int = 768, depth: int = 12, num_tokens: int = 197):
        super().__init__()
        self.embed_dim = dim
        self.num_tokens = num_tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, dim))
        self.proj = nn.Linear(3 * 16 * 16, dim)
        block_list = [_PlaceholderBlock(dim) for _ in range(depth)]
        self.blocks = nn.Sequential(*block_list)
        self.norm = nn.LayerNorm(dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224). Crude patchify -> (B, 196, dim).
        b = x.shape[0]
        patches = F.unfold(x, kernel_size=16, stride=16).transpose(1, 2)
        tokens = self.proj(patches)
        cls = self.cls_token.expand(b, -1, -1)
        z = torch.cat([cls, tokens], dim=1) + self.pos_embed[:, : tokens.shape[1] + 1]
        z = self.blocks(z)
        return self.norm(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)[:, 0]


class _PlaceholderBlock(nn.Module):
    """Stand-in ViT block with the ADAM-style structure."""

    def __init__(
        self, dim: int = 768, mlp_ratio: float = 4.0, num_heads: int = 12
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


# =============================================================================
@dataclass
class SEMAConfig:
    embed_dim: int = 768
    num_layers: int = 12
    expansion_layers: List[int] = field(default_factory=lambda: [9, 10, 11])
    z_threshold: float = 1.0
    adapter_kind: str = "adapter"
    adapter_bottleneck: int = 48
    descriptor_latent: int = 128
    leaky_slope: float = 0.2
    backbone_name: str = "vit_base_patch16_224"
    pretrained: bool = True
    num_classes: int = 0  # prototype classifier; CE head also supported


class SEMA(nn.Module):
    """Top-level SEMA model.

    Public methods:
      * forward(x)               -- classification forward (eval/training).
      * extract_layerwise(x)     -- run backbone collecting layer features.
      * compute_layer_z(x, layer) -- z-scores from existing RDs at `layer`.
      * expand_layer(layer)      -- materialise one new modular adapter slot.
      * register_prototype(class_id, features) -- maintain prototype head.
    """

    def __init__(self, cfg: SEMAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = _try_create_vit(cfg.backbone_name, cfg.pretrained)
        # Freeze backbone (Sec. 3.2).
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.modular_blocks = nn.ModuleDict()
        for l in cfg.expansion_layers:
            self.modular_blocks[str(l)] = ModularAdapterBlock(
                dim=cfg.embed_dim,
                adapter_kind=cfg.adapter_kind,
                adapter_bottleneck=cfg.adapter_bottleneck,
                descriptor_latent=cfg.descriptor_latent,
                leaky_slope=cfg.leaky_slope,
            )
        # Always start with one adapter at every expansion layer (Sec. 3.2 /
        # Appendix A.1: "each transformer block ... is equipped with one
        # adapter module ... at the start of training").
        for l in cfg.expansion_layers:
            self.modular_blocks[str(l)].expand()

        # Prototype classifier (cosine), maintained per class. Shape grows
        # as new classes are seen. We keep a parameter buffer, not a Linear.
        self.register_buffer(
            "prototypes", torch.zeros(0, cfg.embed_dim), persistent=True
        )
        self.register_buffer(
            "prototype_count", torch.zeros(0, dtype=torch.long), persistent=True
        )

    # ------------------------------------------------------------------ utils
    def expansion_layer_set(self) -> List[int]:
        return sorted(int(k) for k in self.modular_blocks.keys())

    def get_block(self, layer_idx: int) -> ModularAdapterBlock:
        return self.modular_blocks[str(layer_idx)]

    # ---------------------------------------------------- forward (training)
    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run input through the (frozen) ViT, applying modular adapters at
        the configured layers via Eq. 3.
        """
        bb = self.backbone

        # --- patchify + cls + positional embedding ---
        if hasattr(bb, "patch_embed"):  # timm path
            tokens = bb.patch_embed(x)
            cls = bb.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1) + bb.pos_embed
            z = bb.pos_drop(tokens) if hasattr(bb, "pos_drop") else tokens
        else:  # placeholder
            patches = F.unfold(x, kernel_size=16, stride=16).transpose(1, 2)
            tokens = bb.proj(patches)
            cls = bb.cls_token.expand(tokens.shape[0], -1, -1)
            z = torch.cat([cls, tokens], dim=1) + bb.pos_embed[:, : tokens.shape[1] + 1]

        # --- pass through transformer blocks, splicing in adapters ---
        for i, block in enumerate(bb.blocks):
            z = self._block_forward_with_adapter(z, block, layer_idx=i)

        if hasattr(bb, "norm"):
            z = bb.norm(z)
        feats = z[:, 0]  # CLS token features

        logits = self._prototype_logits(feats)
        return (logits, feats) if return_features else (logits, None)

    def _block_forward_with_adapter(
        self, x: torch.Tensor, block: nn.Module, layer_idx: int
    ) -> torch.Tensor:
        """Mirror the standard ViT block forward and inject the mixture term.

        Standard ViT block:
            x = x + Attn(LN1(x))
            x = x + MLP(LN2(x))
        SEMA (Eq. 3) adds the adapter mixture in parallel with the MLP:
            x = x + MLP(LN2(x)) + sum_k w_k * f_phi_k(LN2(x))
        """
        # Self-attention residual (untouched by SEMA).
        h = block.norm1(x)
        if hasattr(block, "attn"):
            attn_out = block.attn(h)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            x = x + attn_out
        else:
            x = x + h

        # MLP path with optional adapter mixture.
        h2 = block.norm2(x)
        mlp_out = block.mlp(h2)
        if str(layer_idx) in self.modular_blocks:
            adapter_out = self.modular_blocks[str(layer_idx)](h2)
            return x + mlp_out + adapter_out
        return x + mlp_out

    # ------------------------------------------------------- novelty / probe
    @torch.no_grad()
    def extract_layerwise(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Run the (frozen) backbone and capture pre-MLP features (i.e. the
        input to MLP at each block, which is what RDs are conditioned on).
        Returns a dict mapping layer_idx -> (B, N, D) tensor.

        The MLP path also receives any *frozen* adapter mixture, exactly
        matching the input distribution that future RDs would see.
        """
        bb = self.backbone
        store: Dict[int, torch.Tensor] = {}

        if hasattr(bb, "patch_embed"):
            tokens = bb.patch_embed(x)
            cls = bb.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1) + bb.pos_embed
            z = bb.pos_drop(tokens) if hasattr(bb, "pos_drop") else tokens
        else:
            patches = F.unfold(x, kernel_size=16, stride=16).transpose(1, 2)
            tokens = bb.proj(patches)
            cls = bb.cls_token.expand(tokens.shape[0], -1, -1)
            z = torch.cat([cls, tokens], dim=1) + bb.pos_embed[:, : tokens.shape[1] + 1]

        for i, block in enumerate(bb.blocks):
            h = block.norm1(z)
            if hasattr(block, "attn"):
                a = block.attn(h)
                if isinstance(a, tuple):
                    a = a[0]
                z = z + a
            else:
                z = z + h
            h2 = block.norm2(z)
            if i in self.cfg.expansion_layers:
                store[i] = h2.detach()
            mlp_out = block.mlp(h2)
            if str(i) in self.modular_blocks:
                mlp_out = mlp_out + self.modular_blocks[str(i)](h2)
            z = z + mlp_out

        return store

    @torch.no_grad()
    def compute_layer_z(self, feat: torch.Tensor, layer: int) -> torch.Tensor:
        """Return per-sample max-of-min-over-K z-scores at one layer.

        Following Sec. 3.6: an expansion signal is triggered when *all* RDs
        report z > threshold for a sample. The aggregated indicator we expose
        is min_k z_k^l per sample, so the caller can compare against the
        threshold and aggregate over the batch (e.g. via a fraction).
        """
        block = self.get_block(layer)
        zs = block.z_scores(feat)  # (B, K)
        if zs.shape[-1] == 0:
            return feat.new_zeros(feat.shape[0])
        return zs.min(dim=-1).values

    # --------------------------------------------------------- expansion API
    def expand_layer(self, layer: int) -> int:
        """Add one new modular adapter slot at `layer`. Returns its index."""
        return self.get_block(layer).expand()

    # ------------------------------------------------------ prototype head --
    @torch.no_grad()
    def reset_prototypes(self, num_classes: int) -> None:
        """Allocate / resize the prototype buffer to `num_classes`."""
        self.prototypes = torch.zeros(
            num_classes, self.cfg.embed_dim, device=self.prototypes.device
        )
        self.prototype_count = torch.zeros(
            num_classes, dtype=torch.long, device=self.prototypes.device
        )

    @torch.no_grad()
    def update_prototypes(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        """Incrementally accumulate class means (Sec. 4.2 -- prototype head)."""
        if self.prototypes.shape[0] == 0:
            return
        for c in labels.unique().tolist():
            mask = labels == c
            n_new = int(mask.sum().item())
            if n_new == 0:
                continue
            mean_new = feats[mask].mean(dim=0)
            n_old = int(self.prototype_count[c].item())
            n = n_old + n_new
            self.prototypes[c] = (self.prototypes[c] * n_old + mean_new * n_new) / n
            self.prototype_count[c] = n

    def _prototype_logits(self, feats: torch.Tensor) -> torch.Tensor:
        """Cosine-similarity logits w.r.t. learned prototypes.

        If no prototypes are registered yet, returns logits of shape (B, 0)
        which downstream code interprets as an unfit classifier.
        """
        if self.prototypes.shape[0] == 0:
            return feats.new_zeros(feats.shape[0], 0)
        f = F.normalize(feats, dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        return f @ p.t()

    # --------------------------------------------------- adapter param API --
    def trainable_adapter_parameters(self) -> List[nn.Parameter]:
        """Yield only the parameters that should be optimised this step.

        Frozen adapters and frozen router columns return their grads as 0
        (router uses a hook), so it's safe to include all router params; the
        optimiser will be a no-op on the frozen ones. Old adapters/RDs have
        `requires_grad=False`, so PyTorch will skip them automatically.
        """
        params: List[nn.Parameter] = []
        for block in self.modular_blocks.values():
            for p in block.adapters.parameters():
                if p.requires_grad:
                    params.append(p)
            for p in block.router.parameters():
                if p.requires_grad:
                    params.append(p)
        return params

    def trainable_descriptor_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for block in self.modular_blocks.values():
            for p in block.descriptors.parameters():
                if p.requires_grad:
                    params.append(p)
        return params
