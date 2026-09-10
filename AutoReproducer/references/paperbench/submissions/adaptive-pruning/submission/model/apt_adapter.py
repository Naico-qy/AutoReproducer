"""APT Adapter — Section 4.1 of Zhao et al., ICML 2024.

The APT adapter generalises LoRA (Hu et al. 2022, ICLR) by adding two
binary pruning masks (m_i, m_o) and a *dynamic* low-rank value r_apt.
Equation (2) of the paper:

    H_apt(X) = m_o ∘ ((W + s · W_B W_A) X) ∘ m_i

where:
  * W       : frozen pretrained weight,  shape (d_o, d_i)
  * W_A     : tunable down-projection,   shape (r_apt, d_i)
  * W_B     : tunable up-projection,     shape (d_o, r_apt)
  * m_i     : binary input mask,         shape (d_i,)
  * m_o     : binary output mask,        shape (d_o,)
  * s       : scalar scaling factor (= lora_alpha / r_apt)

Verified citation (CrossRef OK):
    Xia, Zhong, Chen — "Structured Pruning Learns Compact and Accurate
    Models", ACL 2022. doi:10.18653/v1/2022.acl-long.107  (the CoFi
    baseline used by this paper).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class APTAdapter(nn.Module):
    """The low-rank adapter half of an APT layer.

    Holds W_A and W_B and supports growing r_apt while preserving the
    layer's output via LoRA-style zero initialisation of the new rows of
    W_B and Gaussian initialisation of the new rows of W_A (§4.3).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float = 16.0,
        dropout: float = 0.0,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.init_std = init_std
        self.scaling = alpha / max(rank, 1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # W_A : (r, d_i) — Gaussian init (LoRA convention)
        self.W_A = nn.Parameter(torch.zeros(rank, in_features))
        # W_B : (d_o, r) — zero init so initial output = base layer
        self.W_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.normal_(self.W_A, mean=0.0, std=init_std)
        nn.init.zeros_(self.W_B)

    # ------------------------------------------------------------------ #
    def delta_weight(self) -> torch.Tensor:
        """Return s · W_B W_A — the additive low-rank update."""
        return self.scaling * (self.W_B @ self.W_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (..., d_i)
        h = self.drop(x)
        h = F.linear(h, self.W_A)  # (..., r)
        h = F.linear(h, self.W_B)  # (..., d_o)
        return self.scaling * h

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def grow_rank(self, new_rank: int) -> None:
        """Section 4.3: dynamically increase r_apt -> new_rank.

        New rows of W_A are sampled from N(0, σ²); new rows of W_B are
        zero so the layer's output is unchanged at the moment of growth.
        """
        if new_rank <= self.rank:
            return
        extra = new_rank - self.rank
        device = self.W_A.device
        dtype = self.W_A.dtype

        wa_new = torch.empty(extra, self.in_features, device=device, dtype=dtype)
        nn.init.normal_(wa_new, mean=0.0, std=self.init_std)
        wb_new = torch.zeros(self.out_features, extra, device=device, dtype=dtype)

        self.W_A = nn.Parameter(torch.cat([self.W_A.data, wa_new], dim=0))
        self.W_B = nn.Parameter(torch.cat([self.W_B.data, wb_new], dim=1))
        self.rank = new_rank
        self.scaling = self.alpha / max(self.rank, 1)


class APTLinear(nn.Module):
    """Frozen linear W + APT adapter + binary masks (m_i, m_o).

    Replaces an `nn.Linear` from the base PLM (e.g. q_proj, v_proj,
    intermediate.dense, output.dense for RoBERTa).
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features

        # Frozen base weight & bias copied from the pretrained module.
        self.weight = nn.Parameter(
            base.weight.detach().clone(), requires_grad=not freeze_base
        )
        if base.bias is not None:
            self.bias = nn.Parameter(
                base.bias.detach().clone(), requires_grad=not freeze_base
            )
        else:
            self.register_parameter("bias", None)

        # Binary masks. 1 = keep, 0 = pruned. Buffers (not parameters).
        self.register_buffer("m_in", torch.ones(self.in_features))
        self.register_buffer("m_out", torch.ones(self.out_features))

        # APT adapter.
        self.adapter = APTAdapter(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        # Cached activation/grad tensors used by the salience scorer.
        self._cache_in: Optional[torch.Tensor] = None
        self._cache_grad_out: Optional[torch.Tensor] = None
        self._track_stats = False

    # ------------------------------------------------------------------ #
    def enable_tracking(self, flag: bool = True) -> None:
        self._track_stats = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implement Eq. (2):  m_o ∘ ((W + sW_BW_A) x) ∘ m_i
        # Apply input mask first.
        x_masked = x * self.m_in
        # Frozen path.
        out_base = F.linear(x_masked, self.weight, self.bias)
        # Adapter path (low-rank update).
        out_adapt = self.adapter(x_masked)
        out = out_base + out_adapt
        # Output mask.
        out = out * self.m_out

        if self._track_stats and self.training:
            # Activation cache used in salience scoring (sum-along-batch
            # to save memory, as per §4.2 of the paper).
            self._cache_in = (
                x_masked.detach().abs().sum(dim=tuple(range(x_masked.dim() - 1)))
            )
            if out.requires_grad:
                out.register_hook(self._save_grad)
        return out

    def _save_grad(self, grad: torch.Tensor) -> None:
        # Sum |grad| along all but last dim so memory is O(d_o).
        self._cache_grad_out = grad.detach().abs().sum(dim=tuple(range(grad.dim() - 1)))

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """Return the materialised (W + sW_BW_A) ⊙ masks for inference."""
        W = self.weight + self.adapter.delta_weight()
        W = W * self.m_in.unsqueeze(0)
        W = W * self.m_out.unsqueeze(1)
        return W

    @torch.no_grad()
    def set_input_mask(self, mask: torch.Tensor) -> None:
        assert mask.shape == self.m_in.shape
        self.m_in.copy_(mask.to(self.m_in.dtype))

    @torch.no_grad()
    def set_output_mask(self, mask: torch.Tensor) -> None:
        assert mask.shape == self.m_out.shape
        self.m_out.copy_(mask.to(self.m_out.dtype))

    # ------------------------------------------------------------------ #
    @property
    def num_active_in(self) -> int:
        return int(self.m_in.sum().item())

    @property
    def num_active_out(self) -> int:
        return int(self.m_out.sum().item())

    @property
    def num_active_params(self) -> int:
        # Effective parameter count after masks (frozen + adapter).
        a_in = self.num_active_in
        a_out = self.num_active_out
        return a_in * a_out + (a_in + a_out) * self.adapter.rank
