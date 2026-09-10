"""Choosing sigma_max for the VE SDE.

Per the paper (Appendix E.3.1) and Song & Ermon (2020, "Technique 1"), σ_max
is set to the maximum pairwise Euclidean distance among the *training* data
points. The paper's addendum further clarifies:

    "When computing sigma_max for VESDE for sequential methods, only the
     training data points available in the FIRST round should be used."

We therefore compute σ_max once at the end of round 1 (after the first
batch of simulations) and freeze it for all subsequent rounds.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def compute_sigma_max_technique1(theta: torch.Tensor, max_pairs: int = 200000) -> float:
    """Maximum pairwise Euclidean distance among rows of `theta`.

    Following Song & Ermon (2020, "Technique 1"). For very large datasets we
    sub-sample at most `max_pairs` random pairs to keep the computation O(M).
    """
    n = theta.shape[0]
    if n < 2:
        return 1.0

    if n * (n - 1) // 2 <= max_pairs:
        # Full pairwise distance — feasible only for small n.
        diff = theta.unsqueeze(0) - theta.unsqueeze(1)
        dist = torch.norm(diff, dim=-1)
        # Exclude diagonal (self-distance == 0) when taking the max.
        dist.fill_diagonal_(0.0)
        return float(dist.max().item())

    # Sub-sampled estimate.
    idx_a = torch.randint(0, n, (max_pairs,))
    idx_b = torch.randint(0, n, (max_pairs,))
    diff = theta[idx_a] - theta[idx_b]
    dist = torch.norm(diff, dim=-1)
    return float(dist.max().item())
