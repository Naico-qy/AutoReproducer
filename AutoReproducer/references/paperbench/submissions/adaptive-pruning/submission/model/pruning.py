"""Adaptive pruning controller — Section 4.2.

Realises:
  * cubic sparsity schedule γ_t between pruning_start_step and
    pruning_end_step (Zhu & Gupta 2017; the same schedule used by CoFi
    [Xia et al., ACL 2022 — verified DOI 10.18653/v1/2022.acl-long.107]).
  * binary search over the latency-saliency knapsack of MHA heads,
    FFN neurons, and hidden dim (§4.2 'Efficient search').
  * dispatching binary masks back onto each APTLinear layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from .apt_adapter import APTLinear
from .salience import EMASalience, outlier_aware_salience


# --------------------------------------------------------------------------- #
def cubic_sparsity_schedule(
    step: int, start_step: int, end_step: int, target: float, init: float = 0.0
) -> float:
    """γ_t cubic schedule. Returns the *current* sparsity target.

    Equation 1 of Zhu & Gupta '17 (cited in §4.2):
        γ_t = γ_T + (γ_0 - γ_T) (1 - (t-t_b)/(t_e-t_b))^3
    Constant before t_b and after t_e.
    """
    if step < start_step:
        return init
    if step >= end_step:
        return target
    progress = (step - start_step) / max(1, end_step - start_step)
    return target + (init - target) * (1.0 - progress) ** 3


# --------------------------------------------------------------------------- #
@dataclass
class BlockGroup:
    """A pruneable group sharing one mask (head, neuron, or hidden dim)."""

    name: str  # symbolic identifier
    layer_id: int  # transformer layer index (-1 for global hidden dim)
    kind: str  # 'head' | 'neuron' | 'hidden'
    size: int  # number of parameters represented by this group
    score: float  # current outlier-aware salience score


def binary_search_masks(
    groups: List[BlockGroup],
    target_sparsity: float,
    total_params: int,
    max_iters: int = 64,
) -> Dict[str, bool]:
    """Latency-saliency knapsack via binary search (§4.2 / Appendix C).

    The blocks are sorted by `score / size`; we then search for the
    threshold that prunes ≥ `target_sparsity * total_params`.
    """
    if not groups:
        return {}
    # Sort ascending by score/size so the *least* salient density is first.
    order = sorted(groups, key=lambda g: g.score / max(1, g.size))
    # Cumulative parameter count if we *prune* groups[: k].
    cum = []
    s = 0
    for g in order:
        s += g.size
        cum.append(s)
    target_pruned = int(target_sparsity * total_params)

    # Binary-search the smallest k with cum[k-1] >= target_pruned.
    lo, hi = 0, len(order)
    for _ in range(max_iters):
        if lo >= hi:
            break
        mid = (lo + hi) // 2
        if cum[mid] >= target_pruned:
            hi = mid
        else:
            lo = mid + 1
    cutoff = lo

    keep = {g.name: True for g in groups}
    for g in order[:cutoff]:
        keep[g.name] = False
    return keep


# --------------------------------------------------------------------------- #
class PruneController:
    """Coordinates per-step adaptive pruning over the whole APT model."""

    def __init__(
        self,
        target_sparsity: float = 0.6,
        start_step: int = 200,
        end_step: int = 2000,
        ema_decay: float = 0.85,
        use_kurtosis: bool = True,
        prune_heads: bool = True,
        prune_ffn: bool = True,
        prune_hidden: bool = True,
        update_every: int = 50,
    ) -> None:
        self.target_sparsity = target_sparsity
        self.start_step = start_step
        self.end_step = end_step
        self.update_every = update_every
        self.use_kurtosis = use_kurtosis
        self.prune_heads = prune_heads
        self.prune_ffn = prune_ffn
        self.prune_hidden = prune_hidden
        self.ema = EMASalience(decay=ema_decay)
        self._initial_total: int = 0

    # ------------------------------------------------------------------ #
    def reset(self, model: torch.nn.Module) -> None:
        self._initial_total = sum(p.numel() for p in model.parameters())

    def current_target(self, step: int) -> float:
        return cubic_sparsity_schedule(
            step, self.start_step, self.end_step, self.target_sparsity
        )

    # ------------------------------------------------------------------ #
    def _gather_layers(
        self, model: torch.nn.Module
    ) -> Tuple[List[APTLinear], List[APTLinear]]:
        """Return (attention_layers, ffn_layers) APTLinear modules."""
        attn, ffn = [], []
        for name, mod in model.named_modules():
            if not isinstance(mod, APTLinear):
                continue
            lname = name.lower()
            if any(
                k in lname for k in ("query", "value", "q_proj", "v_proj", "k_proj")
            ):
                attn.append((name, mod))
            elif any(
                k in lname
                for k in (
                    "intermediate",
                    "fc1",
                    "fc2",
                    "wi",
                    "wo",
                    "up_proj",
                    "down_proj",
                )
            ):
                ffn.append((name, mod))
        return attn, ffn

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def step(
        self,
        model: torch.nn.Module,
        global_step: int,
        head_dim: int = 64,
        num_heads: int = 12,
    ) -> Dict[str, float]:
        """Update masks once according to the cubic schedule.

        Returns a small dict of statistics for logging.
        """
        if global_step < self.start_step:
            return {"sparsity_target": 0.0}
        if global_step % self.update_every != 0:
            return {"sparsity_target": self.current_target(global_step)}

        target = self.current_target(global_step)
        groups: List[BlockGroup] = []

        attn_layers, ffn_layers = self._gather_layers(model)

        # ---- 1) per-attention-head groups (size ≈ 4 d_h d_m) ----------
        if self.prune_heads:
            for name, mod in attn_layers:
                if mod._cache_in is None or mod._cache_grad_out is None:
                    continue
                score_vec = outlier_aware_salience(
                    mod._cache_in,
                    mod._cache_grad_out,
                    use_kurtosis=self.use_kurtosis,
                )
                score_vec = self.ema.update(f"attn::{name}", score_vec)
                # Group output dim into heads of size head_dim.
                d_o = mod.out_features
                hd = head_dim if d_o % head_dim == 0 else max(1, d_o // num_heads)
                nheads = d_o // hd
                head_scores = score_vec[: nheads * hd].view(nheads, hd).sum(dim=-1)
                for h in range(nheads):
                    groups.append(
                        BlockGroup(
                            name=f"{name}::head{h}",
                            layer_id=-1,
                            kind="head",
                            size=4 * hd * mod.in_features,
                            score=float(head_scores[h]),
                        )
                    )

        # ---- 2) per-FFN-neuron groups --------------------------------
        if self.prune_ffn:
            for name, mod in ffn_layers:
                if mod._cache_in is None or mod._cache_grad_out is None:
                    continue
                score_vec = outlier_aware_salience(
                    mod._cache_in,
                    mod._cache_grad_out,
                    use_kurtosis=self.use_kurtosis,
                )
                score_vec = self.ema.update(f"ffn::{name}", score_vec)
                # Score one entry per output neuron.
                for j in range(score_vec.numel()):
                    groups.append(
                        BlockGroup(
                            name=f"{name}::neuron{j}",
                            layer_id=-1,
                            kind="neuron",
                            size=2 * mod.in_features,
                            score=float(score_vec[j]),
                        )
                    )

        if not groups:
            return {"sparsity_target": target, "n_groups": 0}

        total_params = max(self._initial_total, 1)
        keep = binary_search_masks(groups, target, total_params)

        # ---- 3) write masks back to layers ---------------------------
        new_masks: Dict[str, torch.Tensor] = {}
        for g in groups:
            base, _, idx = g.name.rpartition("::")
            new_masks.setdefault(base, []).append((idx, keep[g.name]))
        # Build per-layer output masks.
        for name, mod in attn_layers + ffn_layers:
            entries = new_masks.get(name)
            if entries is None:
                continue
            d_o = mod.out_features
            mask = torch.ones(d_o, device=mod.m_out.device)
            for idx_str, keep_flag in entries:
                if idx_str.startswith("head"):
                    h = int(idx_str[4:])
                    hd = (
                        d_o // (d_o // 64)
                        if d_o % 64 == 0
                        else max(1, d_o // num_heads)
                    )
                    s, e = h * hd, (h + 1) * hd
                    if not keep_flag:
                        mask[s:e] = 0.0
                elif idx_str.startswith("neuron"):
                    j = int(idx_str[6:])
                    if not keep_flag:
                        mask[j] = 0.0
            mod.set_output_mask(mask)

        active = sum(int(m.m_out.sum().item()) for _, m in attn_layers + ffn_layers)
        total_out = sum(m.out_features for _, m in attn_layers + ffn_layers)
        achieved = 1.0 - (active / max(1, total_out))
        return {
            "sparsity_target": target,
            "sparsity_achieved": achieved,
            "n_groups": len(groups),
        }
