"""Hindsight Experience Replay (HER) goal sampling.

Per Appendix B and the addendum:

    With probability 0.2  -> goal = current state          (terminal/done; r = 0)
    With probability 0.5  -> goal = future state in traj   (geometric)
    With probability 0.3  -> goal = random state in dataset

This same sampler is reused for the GC-IQL and GC-BC baselines, with the
ratios coming directly from the addendum's GC-IQL section.
"""

from __future__ import annotations

import numpy as np
import torch


def sample_her_goals(
    states: torch.Tensor,
    traj_starts: np.ndarray | None,
    batch_idx: torch.Tensor,
    *,
    p_random: float = 0.3,
    p_geometric: float = 0.5,
    p_current: float = 0.2,
    geom_p: float = 1.0 / 50.0,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised HER goal sampler.

    Args:
        states     : (N, state_dim)  full offline buffer
        traj_starts: (T,) indices of trajectory starts (used for geometric sampling)
        batch_idx  : (B,) indices of the current sample within `states`
    Returns:
        goals : (B, state_dim)
        masks : (B,)   1.0 if the goal coincides with the current state
                       (-> reward = 0, terminal flag should be set), else 0.0
    """
    rng = rng or np.random.default_rng()
    b = batch_idx.shape[0]
    n = states.shape[0]
    u = rng.random(b)

    # default: random goal everywhere, then overwrite buckets
    rand_idx = torch.as_tensor(rng.integers(0, n, size=b), dtype=torch.long)

    # geometric future-state sampling
    if traj_starts is None:
        future_idx = torch.minimum(
            batch_idx
            + torch.as_tensor(rng.geometric(geom_p, size=b), dtype=torch.long),
            torch.tensor(n - 1),
        )
    else:
        # find the end of the trajectory containing each batch index
        idx_np = batch_idx.cpu().numpy()
        # binary search for "next trajectory start > idx_np"
        starts = np.asarray(traj_starts)
        ends = np.searchsorted(starts, idx_np, side="right")
        ends = np.where(
            ends < len(starts), starts[np.minimum(ends, len(starts) - 1)] - 1, n - 1
        )
        offsets = rng.geometric(geom_p, size=b)
        fut = np.minimum(idx_np + offsets, ends)
        future_idx = torch.as_tensor(fut, dtype=torch.long)

    use_current = u < p_current
    use_geom = (u >= p_current) & (u < p_current + p_geometric)
    final = torch.where(
        torch.as_tensor(use_current),
        batch_idx,
        torch.where(torch.as_tensor(use_geom), future_idx, rand_idx),
    )

    goals = states[final]
    mask = torch.as_tensor(use_current, dtype=torch.float32)
    return goals, mask
