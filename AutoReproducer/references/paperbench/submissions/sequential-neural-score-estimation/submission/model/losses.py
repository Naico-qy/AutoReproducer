"""Denoising posterior score-matching loss for (T)SNPSE.

Implements Eq. (7) of Sharrock et al., ICML 2024:

    J_post^DSM(ψ) = ½ ∫_0^T λ_t · E_{p_{t|0}(θ_t|θ_0) p(x|θ_0) p(θ_0)}[
                       ‖ s_ψ(θ_t, x, t) − ∇_{θ_t} log p_{t|0}(θ_t | θ_0) ‖²
                    ] dt.

For TSNPSE we draw θ_0 from the round-r proposal prior p̃^r(θ); under the
support condition of Proposition 3.1 (paper Section 3.1), this still recovers
the true posterior score, so the same loss form applies.

The target ``∇_{θ_t} log p_{t|0}(θ_t | θ_0)`` is computed in closed form by the
SDE (`BaseSDE.add_noise`).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .sde import BaseSDE


def denoising_score_matching_loss(
    score_net: nn.Module,
    sde: BaseSDE,
    theta: torch.Tensor,
    x: torch.Tensor,
    t_eps: float = 1e-5,
    likelihood_weighting: bool = False,
    importance_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute a Monte-Carlo estimate of the DSM loss in Eq. (7) of the paper.

    Parameters
    ----------
    score_net : score network ``s_ψ(θ_t, x, t)``.
    sde       : forward SDE (VESDE or VPSDE).
    theta     : (B, d) tensor sampled from p(θ) (or proposal prior p̃^r(θ)).
    x         : (B, p) simulated data sampled from p(x | θ).
    t_eps     : small lower bound on t to avoid singularities at t=0.
    likelihood_weighting : if True, multiply the squared error by g(t)² (an
        alternative weighting `λ_t = g(t)²` proposed by Song et al. 2021,
        which is closely related to maximum-likelihood training).
    importance_weights : optional (B,) tensor of importance weights, used by
        the SNPSE-B variant (Eq. 15 of the paper) to correct for proposal
        prior bias. When ``None`` the weights default to 1 (NPSE / TSNPSE).

    Returns
    -------
    loss : scalar tensor.
    """
    batch_size = theta.shape[0]
    device = theta.device

    # Sample diffusion times uniformly in [t_eps, T].
    t = torch.rand(batch_size, device=device) * (sde.T - t_eps) + t_eps

    # Sample θ_t and the score target from the SDE transition density.
    theta_t, score_target, std = sde.add_noise(theta, t)

    # Predict the score with the network.
    score_pred = score_net(theta_t, x, t)

    # Squared L2 error per sample, summed over feature dim.
    sq_err = ((score_pred - score_target) ** 2).sum(dim=-1)

    # Weighting:  λ_t = σ_t² (default) so that the loss equals the standard
    # noise-prediction objective; this is the choice used in Song et al. 2021
    # and is consistent with the paper's Appendix A.1 derivation.
    if std.ndim > 1:
        std = std.squeeze(-1)
    weights = std**2
    weighted = weights * sq_err

    if likelihood_weighting:
        g = sde.diffusion(t)
        weighted = (g**2) * sq_err

    if importance_weights is not None:
        # SNPSE-B (Eq. 15): multiply per-sample loss by  p(θ_0) / p̃^r(θ_0).
        weighted = weighted * importance_weights

    return weighted.mean()
