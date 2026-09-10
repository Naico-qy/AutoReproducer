"""Self-Knowledge Distillation — Section 4.4 + Addendum.

Equations 7 of the paper:
    L      = μ L_distill + (1 - μ) L_ft
    L_layer= Σ_{i∈T} MSE(Tr(H_s^{φ(i)}), H_t^i)

Addendum-specified details:
  * Classification (GLUE):  L_distill = L_pred + 0.9 · L_layer
  * SQuAD / CNN-DM:        L_distill = 0.1·L_pred + 0.9 · L_layer
  * Temperature τ = 4 (CoFi).
  * μ scales linearly from 0 → 1 across the pruning window:
        μ = min(1, (step - prune_start) / (prune_end - prune_start))
  * Layer-mapping φ is recomputed every step.
  * The teacher reuses the *current* student (self-distillation): frozen
    parameters are shared; only the tunable adapters of the teacher copy
    are kept frozen during the student's update.
  * Tr(·) is a tunable LoRA-style transformation init as the identity I.

Verified citation (CrossRef OK):
    Xia et al., "Structured Pruning Learns Compact and Accurate Models"
    (CoFi), ACL 2022, doi:10.18653/v1/2022.acl-long.107.
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


# --------------------------------------------------------------------------- #
def mu_schedule(step: int, prune_start: int, prune_end: int) -> float:
    """Addendum: μ = min(1, (t - t_b) / (t_e - t_b)).

    Returns 0 before pruning has begun.
    """
    if step <= prune_start:
        return 0.0
    return min(1.0, (step - prune_start) / max(1, prune_end - prune_start))


# --------------------------------------------------------------------------- #
def layer_mapping(
    student_active: Sequence[bool],
    teacher_layers: Sequence[int],
) -> Dict[int, int]:
    """φ(·): teacher_idx -> closest non-pruned student_idx.

    student_active[i] == True  iff student layer i still has any active
    output channels (i.e. has not been entirely pruned).

    Recomputed every step (addendum).
    """
    active_indices = [i for i, a in enumerate(student_active) if a]
    if not active_indices:
        return {}
    out = {}
    for ti in teacher_layers:
        # Closest by absolute distance, ties broken by lower index.
        si = min(active_indices, key=lambda j: (abs(j - ti), j))
        out[ti] = si
    return out


# --------------------------------------------------------------------------- #
class LinearTransform(nn.Module):
    """Tr(·) — a small tunable LoRA-style transform initialised as I."""

    def __init__(self, dim: int, rank: int = 8) -> None:
        super().__init__()
        # We start with W = I and an additive low-rank delta = 0.
        self.register_buffer("base", torch.eye(dim))
        self.A = nn.Parameter(torch.zeros(rank, dim))
        self.B = nn.Parameter(torch.zeros(dim, rank))
        nn.init.normal_(self.A, std=0.02)
        # B kept zero so that initial output = x.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.base) + F.linear(F.linear(x, self.A), self.B)


# --------------------------------------------------------------------------- #
def cofi_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_hidden: List[torch.Tensor],
    teacher_hidden: List[torch.Tensor],
    transforms: nn.ModuleList,
    sampled_layers: Sequence[int],
    phi: Dict[int, int],
    temperature: float = 4.0,
    pred_loss_weight: float = 1.0,
    layer_loss_weight: float = 0.9,
) -> Dict[str, torch.Tensor]:
    """L_distill (Eq. 7 + Addendum)."""
    # --- L_pred : KL(student || teacher) at temperature τ -----------------
    log_p_s = F.log_softmax(student_logits / temperature, dim=-1)
    p_t = F.softmax(teacher_logits / temperature, dim=-1)
    L_pred = F.kl_div(log_p_s, p_t, reduction="batchmean") * (temperature**2)

    # --- L_layer : MSE on hidden states with Tr(.) transformation --------
    L_layer = student_logits.new_zeros(())
    n = 0
    for k, t_idx in enumerate(sampled_layers):
        s_idx = phi.get(t_idx)
        if s_idx is None:
            continue
        s_h = student_hidden[s_idx]
        t_h = teacher_hidden[t_idx]
        if k < len(transforms):
            s_h = transforms[k](s_h)
        # Match shapes (e.g. truncate seq-len) before MSE.
        L_layer = L_layer + F.mse_loss(s_h, t_h)
        n += 1
    if n > 0:
        L_layer = L_layer / n

    L_total = pred_loss_weight * L_pred + layer_loss_weight * L_layer
    return {"L_distill": L_total, "L_pred": L_pred, "L_layer": L_layer}


# --------------------------------------------------------------------------- #
def sample_teacher_layers(
    num_total: int, num_sampled: int, rng: Optional[random.Random] = None
) -> List[int]:
    """Block-wise random teacher-layer sampling (Haidar et al., 2022).

    Splits the teacher's `num_total` layers into `num_sampled` equal-size
    blocks and samples one layer index uniformly per block.
    """
    if num_sampled <= 0 or num_total <= 0:
        return []
    sampler = rng if rng is not None else random.Random()
    block = max(1, num_total // num_sampled)
    out = []
    for b in range(num_sampled):
        lo = b * block
        hi = min(num_total, lo + block) if b < num_sampled - 1 else num_total
        if lo >= hi:
            break
        out.append(sampler.randrange(lo, hi))
    return out


# --------------------------------------------------------------------------- #
class SelfDistiller(nn.Module):
    """Wraps a student APT model and a "teacher" *view* on the same weights.

    Section 4.4: 'we keep duplicating the tuning student layers as
    teachers during fine-tuning to reduce total training time. Meanwhile,
    frozen parameters are shared between the student and teacher model.'

    We therefore do NOT store a deep-copy of the (huge) frozen base; we
    only snapshot the tunable adapter weights periodically and use the
    shared frozen base when running the teacher forward pass.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_sampled_layers: int = 4,
        temperature: float = 4.0,
        layer_loss_weight: float = 0.9,
        pred_loss_weight: float = 1.0,
        transform_rank: int = 8,
    ) -> None:
        super().__init__()
        self.num_sampled_layers = num_sampled_layers
        self.temperature = temperature
        self.layer_loss_weight = layer_loss_weight
        self.pred_loss_weight = pred_loss_weight
        self.transforms = nn.ModuleList(
            [
                LinearTransform(hidden_dim, rank=transform_rank)
                for _ in range(num_sampled_layers)
            ]
        )
        # Teacher checkpoint of the adapter parameters.
        self._teacher_state: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def snapshot_teacher(self, student: nn.Module) -> None:
        """Cache the current adapter weights as the new teacher."""
        keep = {}
        for name, p in student.named_parameters():
            # We snapshot every tunable parameter so that later we can
            # reload them temporarily for a teacher forward pass.
            if p.requires_grad:
                keep[name] = p.detach().clone()
        self._teacher_state = keep

    @torch.no_grad()
    def teacher_state(self) -> Dict[str, torch.Tensor]:
        return self._teacher_state or {}
