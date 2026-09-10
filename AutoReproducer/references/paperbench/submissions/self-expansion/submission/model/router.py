"""Expandable Weighting Router for SEMA (Sec. 3.4, Eq. 3).

For any layer l with K^l adapters,

    w^l = h_psi(x^l) = softmax(x^l . W_mix^l)             (Eq. 3)
    x_out^l = MLP(x^l) + sum_k w_k^l * f_phi_k(x^l)

When a new adapter is added at layer l, the router is expanded by appending
one new column to W_mix and only that column is trainable; the previously
trained columns are frozen to prevent forgetting on routing (Sec. 3.4).

Following [Dou et al., 2023 -- LoRAMoE], h_psi is implemented as a single
linear layer followed by softmax. We store the mixture matrix as an
nn.Parameter and rebuild it whenever the router is expanded.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ExpandableRouter(nn.Module):
    """Linear+softmax router whose output dimension grows with K^l."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        # Start with K=0 experts; columns are added by `expand_one_slot`.
        self.W_mix = nn.Parameter(torch.zeros(dim, 0), requires_grad=False)
        # Track which columns are trainable (only the most recent).
        self.register_buffer(
            "trainable_mask", torch.zeros(0, dtype=torch.bool), persistent=False
        )

    # --------------------------------------------------------------- expand
    @torch.no_grad()
    def expand_one_slot(self) -> int:
        """Append one new column (initialised with kaiming_uniform).

        Returns the index of the new column. Previous columns are frozen by
        clearing their entries in `trainable_mask`.
        """
        new_col = torch.empty(
            self.dim, 1, device=self.W_mix.device, dtype=self.W_mix.dtype
        )
        nn.init.kaiming_uniform_(new_col, a=math.sqrt(5))
        new_W = torch.cat([self.W_mix.detach(), new_col], dim=1)
        # Replace parameter while preserving gradient graph compatibility.
        self.W_mix = nn.Parameter(new_W, requires_grad=True)
        # Freeze all but the newest column.
        new_mask = torch.zeros(new_W.shape[1], dtype=torch.bool, device=new_W.device)
        new_mask[-1] = True
        self.trainable_mask = new_mask
        return new_W.shape[1] - 1

    # --------------------------------------------------------------- helpers
    @property
    def num_experts(self) -> int:
        return self.W_mix.shape[1]

    def parameters_for_optimizer(self):
        """Yield only the trainable column slice (newest expert).

        We achieve column-wise freezing via a backward hook that zeros out
        gradients on frozen columns; `parameters_for_optimizer` simply yields
        `self.W_mix` so that the optimiser can update only the live slot.
        """
        if self.W_mix.shape[1] == 0:
            return iter(())
        return iter([self.W_mix])

    def freeze_old_columns_grad(self) -> None:
        """Register a backward hook that zeros gradients on frozen columns."""
        if self.W_mix.shape[1] == 0:
            return
        mask = self.trainable_mask.float().unsqueeze(0)  # (1, K)

        def hook(grad: torch.Tensor) -> torch.Tensor:
            return grad * mask.to(grad.device, grad.dtype)

        # Remove any previous hooks by re-registering on a fresh parameter.
        self.W_mix.register_hook(hook)

    # --------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax mixture weights w^l of shape (..., K^l).

        If K^l == 0 (no adapters yet at this layer), returns a tensor with
        last dimension 0 -- the caller should treat this as a no-op branch.
        """
        if self.W_mix.shape[1] == 0:
            shape = list(x.shape[:-1]) + [0]
            return x.new_zeros(shape)
        logits = x @ self.W_mix
        return F.softmax(logits, dim=-1)
