"""Adaptive tuning controller — Section 4.3.

Maintains the rank budget Δ_t and dynamically grows the rank r_apt of the
top-half most-salient APT adapters following:

    r_apt' = floor( r_apt · Δ_{t'} / Δ_t )

When new rows of W_A and columns of W_B are appended, W_A receives
N(0, σ²) initialisation and W_B receives zeros (LoRA-style) so that the
adapter's output is unchanged at the moment of growth (§4.3).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .apt_adapter import APTAdapter, APTLinear
from .salience import adapter_salience


class RankController:
    """Schedules adapter-rank growth based on salience importance."""

    def __init__(
        self,
        init_rank: int,
        max_rank: int,
        growth_steps: List[int],
        rank_budget_schedule: List[float],
        topk_fraction: float = 0.5,
    ) -> None:
        if len(rank_budget_schedule) != len(growth_steps) + 1:
            raise ValueError(
                "rank_budget_schedule must have len(growth_steps)+1 entries "
                f"(got {len(rank_budget_schedule)} vs {len(growth_steps)})"
            )
        self.init_rank = init_rank
        self.max_rank = max_rank
        self.growth_steps = sorted(growth_steps)
        self.budget = rank_budget_schedule
        self.topk_fraction = topk_fraction
        self._stage = 0

    # ------------------------------------------------------------------ #
    def current_budget(self, step: int) -> float:
        stage = 0
        for i, s in enumerate(self.growth_steps):
            if step >= s:
                stage = i + 1
        return self.budget[min(stage, len(self.budget) - 1)]

    def should_grow(self, step: int) -> Optional[int]:
        """Return new stage index if step crosses a growth boundary."""
        for i, s in enumerate(self.growth_steps):
            if step == s and i + 1 > self._stage:
                self._stage = i + 1
                return i + 1
        return None

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def grow(self, model: torch.nn.Module, step: int) -> Dict[str, int]:
        """Grow ranks of top-fraction salient adapters.

        Eq.: r_apt' = ⌊ r_apt · Δ_{t'} / Δ_t ⌋

        Returns a dict {layer_name: new_rank}.
        """
        new_stage = self.should_grow(step)
        if new_stage is None:
            return {}
        prev_budget = self.budget[new_stage - 1]
        next_budget = self.budget[new_stage]
        ratio = next_budget / max(prev_budget, 1e-9)
        if ratio <= 1.0:
            return {}

        # 1. Score every APT adapter.
        scored: List[tuple] = []
        for name, mod in model.named_modules():
            if isinstance(mod, APTLinear):
                scored.append((name, mod, adapter_salience(mod.adapter)))
        if not scored:
            return {}
        scored.sort(key=lambda t: t[2], reverse=True)
        k = max(1, int(len(scored) * self.topk_fraction))
        top = scored[:k]

        out: Dict[str, int] = {}
        for name, mod, _ in top:
            old_r = mod.adapter.rank
            new_r = min(self.max_rank, int(old_r * ratio))
            if new_r > old_r:
                mod.adapter.grow_rank(new_r)
                out[name] = new_r
        return out

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def total_tunable_params(self, model: torch.nn.Module) -> int:
        s = 0
        for mod in model.modules():
            if isinstance(mod, APTAdapter):
                s += mod.W_A.numel() + mod.W_B.numel()
        return s
