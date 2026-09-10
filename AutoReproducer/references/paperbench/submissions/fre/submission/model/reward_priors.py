"""Random reward function families used as the FRE prior p(eta).

Implements the three reward families described in Section 4.2 / Appendix B
of the paper, plus convenience samplers for the FRE-* mixtures defined in
the addendum.

Each reward function is a callable `eta : (B, state_dim) -> (B,)` that
operates on torch tensors.  Reward functions also expose useful metadata
(e.g. the goal state, MLP weights) so that we can compute true rewards on
arbitrary states later (e.g. during policy training).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


# ---------------------------------------------------------------------------
# 1) Singleton goal-reaching rewards
# ---------------------------------------------------------------------------
@dataclass
class GoalReward:
    """r(s) = 0 if ||s - g|| < threshold else -1.

    `g` is sampled from the offline dataset; `mask_dims` lets AntMaze use
    only the X,Y coordinates (Appendix C.1).
    """

    goal: torch.Tensor  # (state_dim,) or (xy_dim,)
    threshold: float = 2.0
    mask_dims: Optional[slice] = None
    done_on_reach: bool = True

    def __call__(self, s: torch.Tensor) -> torch.Tensor:
        if self.mask_dims is not None:
            sm = s[..., self.mask_dims]
            gm = self.goal
        else:
            sm = s
            gm = self.goal
        d = torch.linalg.vector_norm(sm - gm, dim=-1)
        # reward in {-1, 0}; +0 only when "achieved".
        return torch.where(d < self.threshold, torch.zeros_like(d), -torch.ones_like(d))

    def done(self, s: torch.Tensor) -> torch.Tensor:
        if self.mask_dims is not None:
            sm = s[..., self.mask_dims]
        else:
            sm = s
        d = torch.linalg.vector_norm(sm - self.goal, dim=-1)
        return (d < self.threshold).float()


# ---------------------------------------------------------------------------
# 2) Random LINEAR rewards : r(s) = w . s,  w ~ U(-1,1)^d, sparse mask 0.9
# ---------------------------------------------------------------------------
@dataclass
class LinearReward:
    weight: torch.Tensor  # (state_dim,)
    bias: float = 0.0

    def __call__(self, s: torch.Tensor) -> torch.Tensor:
        return s @ self.weight + self.bias


def sample_linear_reward(
    state_dim: int,
    *,
    sparsity: float = 0.9,
    skip_dims: Optional[slice] = None,
    device: str | torch.device = "cpu",
) -> LinearReward:
    """Per Appendix B: w ~ U(-1, 1)^d, then a binary mask zeros each entry
    with probability `sparsity` (= 0.9) to encourage simple/sparse rewards.

    `skip_dims` zeros specified state dims unconditionally -- used in AntMaze
    to remove XY position dims (whose scale destabilises training).
    """
    w = torch.empty(state_dim, device=device).uniform_(-1.0, 1.0)
    keep = (torch.rand(state_dim, device=device) > sparsity).float()
    w = w * keep
    if skip_dims is not None:
        w[skip_dims] = 0.0
    return LinearReward(weight=w)


# ---------------------------------------------------------------------------
# 3) Random MLP rewards : r(s) = clip(MLP_2(s), -1, 1)
# ---------------------------------------------------------------------------
class RandomMLPReward(torch.nn.Module):
    """Random 2-layer MLP (state_dim -> 32 -> 1) with weights drawn from
    N(0, 1/sqrt(fan_in)) (per Appendix B "scaled by the average dimension
    of the layer"); tanh activation; output clipped to [-1, 1]."""

    def __init__(self, state_dim: int, hidden: int = 32):
        super().__init__()
        self.l1 = torch.nn.Linear(state_dim, hidden)
        self.l2 = torch.nn.Linear(hidden, 1)
        with torch.no_grad():
            self.l1.weight.normal_(0.0, 1.0 / max(state_dim, 1) ** 0.5)
            self.l2.weight.normal_(0.0, 1.0 / max(hidden, 1) ** 0.5)
            self.l1.bias.zero_()
            self.l2.bias.zero_()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.l1(s))
        return torch.clamp(self.l2(h).squeeze(-1), -1.0, 1.0)


def sample_mlp_reward(
    state_dim: int, *, hidden: int = 32, device: str | torch.device = "cpu"
) -> RandomMLPReward:
    return RandomMLPReward(state_dim, hidden).to(device)


# ---------------------------------------------------------------------------
# Mixture sampler (FRE-all / FRE-goals / etc, per addendum)
# ---------------------------------------------------------------------------
def sample_reward(
    prior: str,
    state_dim: int,
    *,
    goal_pool: Optional[torch.Tensor] = None,
    goal_threshold: float = 2.0,
    goal_mask: Optional[slice] = None,
    linear_skip: Optional[slice] = None,
    device: str | torch.device = "cpu",
) -> Callable:
    """Sample one reward function from the requested FRE-* mixture.

    `goal_pool` : (N, state_dim) -- pool of dataset states from which to
                  sample goal-reaching targets.
    """
    families: list[str]
    if prior == "fre-all":
        families = ["goal", "lin", "mlp"]
    elif prior == "fre-goals":
        families = ["goal"]
    elif prior == "fre-lin":
        families = ["lin"]
    elif prior == "fre-mlp":
        families = ["mlp"]
    elif prior == "fre-lin-mlp":
        families = ["lin", "mlp"]
    elif prior == "fre-goal-mlp":
        families = ["goal", "mlp"]
    elif prior == "fre-goal-lin":
        families = ["goal", "lin"]
    else:
        raise ValueError(f"unknown prior: {prior}")

    choice = families[torch.randint(len(families), (1,)).item()]
    if choice == "goal":
        assert goal_pool is not None, "goal_pool required for goal-reaching prior"
        idx = torch.randint(0, goal_pool.shape[0], (1,)).item()
        g = goal_pool[idx]
        if goal_mask is not None:
            g = g[goal_mask]
        return GoalReward(
            goal=g.to(device), threshold=goal_threshold, mask_dims=goal_mask
        )
    if choice == "lin":
        return sample_linear_reward(state_dim, skip_dims=linear_skip, device=device)
    if choice == "mlp":
        return sample_mlp_reward(state_dim, device=device)
    raise RuntimeError("unreachable")
