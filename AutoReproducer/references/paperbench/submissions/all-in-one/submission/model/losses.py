"""Simformer training loss (§3.3, Eq. 6 / Eq. 7 in the paper).

We sample t ~ U(eps, T) and a per-batch-element condition mask M_C from a
mixture of {joint, posterior, likelihood, Bernoulli(0.3), Bernoulli(0.7)}
following addendum.md.

The (partially) noised input is

    x̂_t^{M_C} = (1 - M_C) ⊙ x̂_t + M_C ⊙ x̂_0 ,

i.e. variables we condition on remain clean. The loss is the masked
denoising score-matching objective:

    ℓ = (1 - M_C) ⊙ ( s_φ^{M_E}(x̂_t^{M_C}, t)
                       - ∇_{x̂_t} log p_t(x̂_t | x̂_0) ) ,
    L(φ) = E[ ‖ℓ‖² ] .

For Gaussian transition kernels p_t(x_t | x_0) = N(mean(x_0,t), std(t)²I)
the conditional score is  -(x_t - mean) / std² .
"""

from __future__ import annotations

import torch
from torch import Tensor

from .sde import BaseSDE


# ---------------------------------------------------------------------------
# Mask sampling (addendum.md, training section)
# ---------------------------------------------------------------------------


def sample_condition_mask(
    batch_size: int, num_params: int, num_data: int, probs: dict, device
) -> Tensor:
    """Return a (B, N) boolean condition mask M_C."""
    N = num_params + num_data
    keys = ["joint", "posterior", "likelihood", "bernoulli_0p3", "bernoulli_0p7"]
    weights = torch.tensor([float(probs.get(k, 0.0)) for k in keys], device=device)
    weights = weights / weights.sum().clamp_min(1e-8)
    cats = torch.multinomial(weights, batch_size, replacement=True)
    mask = torch.zeros(batch_size, N, dtype=torch.bool, device=device)
    for b in range(batch_size):
        kind = keys[int(cats[b].item())]
        if kind == "joint":
            pass  # all False
        elif kind == "posterior":
            # parameters latent (False), data conditioned (True)
            mask[b, num_params:] = True
        elif kind == "likelihood":
            # data latent (False), parameters conditioned (True)
            mask[b, :num_params] = True
        elif kind == "bernoulli_0p3":
            mask[b] = torch.rand(N, device=device) < 0.3
        elif kind == "bernoulli_0p7":
            mask[b] = torch.rand(N, device=device) < 0.7
    return mask


# ---------------------------------------------------------------------------
# Time sampling
# ---------------------------------------------------------------------------


def sample_t(batch_size: int, sde: BaseSDE, device) -> Tensor:
    return torch.rand(batch_size, device=device) * (sde.T - sde.eps) + sde.eps


# ---------------------------------------------------------------------------
# Loss (Eq. 7)
# ---------------------------------------------------------------------------


def simformer_loss(
    model,
    x0: Tensor,
    sde: BaseSDE,
    num_params: int,
    mask_probs: dict,
    attention_mask: Tensor | None = None,
    reduce: bool = True,
) -> Tensor:
    """Compute the Simformer denoising-score-matching loss.

    Parameters
    ----------
    model           : the Simformer score network
    x0              : (B, N) clean joint samples (theta, x)
    sde             : the SDE
    num_params      : dimensionality of theta in the joint vector
    mask_probs      : weights for the M_C mixture
    attention_mask  : (N, N) or (B, N, N) bool M_E (True = block); None = dense
    reduce          : whether to return scalar mean
    """
    B, N = x0.shape
    device = x0.device
    num_data = N - num_params

    t = sample_t(B, sde, device)
    M_C = sample_condition_mask(B, num_params, num_data, mask_probs, device)

    # Noised joint
    x_t, mean, std = sde.perturb(x0, t)  # (B, N), (B, N), (B, N)

    # Conditional score of the Gaussian transition kernel
    target = -(x_t - mean) / (std**2)  # (B, N)

    # Replace conditioned coordinates with their clean values (paper §3.3).
    x_in = torch.where(M_C, x0, x_t)

    score = model(x_in, M_C, t, attention_mask)  # (B, N)

    # λ(t) weighting: standard choice λ(t) = std(t)^2 (cancels the noise scale)
    lam = std**2

    diff = score - target
    # Only the latent coordinates contribute to the loss (per Eq. 6).
    weighted = (~M_C).float() * lam * diff**2
    if reduce:
        # Mean over batch and latent coordinates
        denom = (~M_C).float().sum().clamp_min(1.0)
        return weighted.sum() / denom
    return weighted
