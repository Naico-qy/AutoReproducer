"""ModularAdapterBlock: per-layer container of adapters + RDs + router.

Each ViT layer that allows expansion holds one of these. It manages the lists
of adapters (f_phi_k) and representation descriptors (g_varphi_k) at layer l,
together with the expandable weighting router h_psi^l (Sec. 3.3 / 3.4).

Forward path (Eq. 3):
    x_out^l = MLP(x^l) + sum_k w_k^l * f_phi_k(x^l)

The MLP forward is performed by the parent ViT block; this container only
contributes the adapter mixture term, which is added to the MLP output.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn

from .adapters import build_adapter
from .descriptor import RepresentationDescriptor
from .router import ExpandableRouter


class ModularAdapterBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        adapter_kind: str = "adapter",
        adapter_bottleneck: int = 48,
        descriptor_latent: int = 128,
        leaky_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.adapter_kind = adapter_kind
        self.adapter_bottleneck = adapter_bottleneck
        self.descriptor_latent = descriptor_latent
        self.leaky_slope = leaky_slope

        self.adapters: nn.ModuleList = nn.ModuleList()
        self.descriptors: nn.ModuleList = nn.ModuleList()
        self.router = ExpandableRouter(dim=dim)

    # ---------------------------------------------------------------- expand
    def expand(self) -> int:
        """Add one new adapter, RD, and router slot. Returns the new index."""
        adapter = build_adapter(
            self.adapter_kind, dim=self.dim, bottleneck=self.adapter_bottleneck
        )
        descriptor = RepresentationDescriptor(
            dim=self.dim,
            latent_dim=self.descriptor_latent,
            leaky_slope=self.leaky_slope,
        )
        self.adapters.append(adapter)
        self.descriptors.append(descriptor)
        self.router.expand_one_slot()
        # Freeze previously trained adapters/descriptors (Sec. 3.5).
        for i in range(len(self.adapters) - 1):
            for p in self.adapters[i].parameters():
                p.requires_grad_(False)
            for p in self.descriptors[i].parameters():
                p.requires_grad_(False)
        # Tell the router to zero gradients on frozen columns.
        self.router.freeze_old_columns_grad()
        return len(self.adapters) - 1

    @property
    def K(self) -> int:
        return len(self.adapters)

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mixture branch sum_k w_k * f_phi_k(x)."""
        if self.K == 0:
            return torch.zeros_like(x)
        # Compute pooled token (mean of patch tokens excluding CLS for routing)
        # to obtain a per-sample weight vector w of shape (B, K).
        if x.dim() == 3:
            pool = x.mean(dim=1)  # (B, D)
        else:
            pool = x
        w = self.router(pool)  # (B, K)

        # Stack adapter outputs.
        outs: List[torch.Tensor] = [adapter(x) for adapter in self.adapters]
        stacked = torch.stack(outs, dim=0)  # (K, B, ..., D)

        if x.dim() == 3:
            # broadcast w (B, K) to (K, B, 1, 1)
            w_b = w.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
        else:
            w_b = w.transpose(0, 1).unsqueeze(-1)
        mixed = (stacked * w_b).sum(dim=0)
        return mixed

    # --------------------------------------------------------- novelty signal
    @torch.no_grad()
    def z_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, K) z-scores from each existing RD on input features.

        If K == 0, returns an empty tensor.
        """
        if self.K == 0:
            return x.new_zeros(x.shape[0], 0)
        feats = x.mean(dim=1) if x.dim() == 3 else x
        zs = [d.z_score(feats) for d in self.descriptors]
        return torch.stack(zs, dim=-1)  # (B, K)
