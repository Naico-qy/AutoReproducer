"""TSNPSE — truncated proposal computation (paper Section 3.1, Algorithm 1, Appendix E.3.3).

The truncated proposal is

    p̄^r(θ) ∝ p(θ) · 1{ θ ∈ HPR_ε( p_ψ^{r-1}(θ | x_obs) ) }                (Eq. 9)

and the round-r proposal is the mixture

    p̃^r(θ) = (1/r) Σ_{s=0}^{r-1} p̄^s(θ),  with p̄^0(θ) = p(θ).

To compute HPR_ε in practice (Appendix E.3.3) we:

  1. draw 20000 samples θ̃_i ~ p_ψ^{r-1}(· | x_obs) by integrating the
     probability-flow ODE with the current score network;
  2. compute ℓ_i = log p_ψ^{r-1}(θ̃_i | x_obs) via the instantaneous
     change-of-variables formula;
  3. set the truncation threshold κ = quantile_ε({ℓ_i}), with ε = 5e-4 by
     default (paper Appendix E.3.3).

To then sample the truncated proposal we use rejection sampling on the
prior. To minimise expensive likelihood evaluations we first reject any
prior sample lying outside the empirical hypercube of the posterior samples
(paper Appendix E.3.3).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from .sampler import log_prob_probability_flow, sample_probability_flow
from .sde import BaseSDE
from torch import nn


class TruncatedProposal:
    """Truncated proposal p̄^r(θ) for round r of TSNPSE."""

    def __init__(
        self,
        score_net: nn.Module,
        sde: BaseSDE,
        x_obs: torch.Tensor,
        prior_sampler: Callable[[int], torch.Tensor],
        prior_log_prob: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        n_posterior_samples: int = 20000,
        epsilon: float = 5e-4,
        device: Optional[torch.device] = None,
    ):
        self.score_net = score_net
        self.sde = sde
        self.x_obs = x_obs
        self.prior_sampler = prior_sampler
        self.prior_log_prob = prior_log_prob
        self.n_posterior_samples = n_posterior_samples
        self.epsilon = float(epsilon)
        self.device = device or next(score_net.parameters()).device

        self._kappa: Optional[float] = None
        self._bbox_low: Optional[torch.Tensor] = None
        self._bbox_high: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def fit(self) -> None:
        """Compute the threshold κ and the empirical bounding box."""
        self.score_net.eval()
        post = sample_probability_flow(
            self.score_net,
            self.sde,
            self.x_obs.to(self.device),
            n_samples=self.n_posterior_samples,
            method="rk45",
            device=self.device,
        )
        self._bbox_low = post.min(dim=0).values.detach()
        self._bbox_high = post.max(dim=0).values.detach()

        log_p = log_prob_probability_flow(
            self.score_net,
            self.sde,
            post,
            self.x_obs.to(self.device),
            hutchinson=True,
            n_eps=1,
        )
        # κ = ε-quantile of the log-densities of the approximate posterior samples.
        self._kappa = float(torch.quantile(log_p, self.epsilon).item())

    # ------------------------------------------------------------------
    def in_hpr(self, theta: torch.Tensor) -> torch.Tensor:
        """Return a bool mask of theta in the highest-probability region."""
        if self._kappa is None:
            raise RuntimeError("TruncatedProposal.fit() must be called first.")
        # Cheap pre-rejection by bounding box (paper Appendix E.3.3).
        in_box = ((theta >= self._bbox_low) & (theta <= self._bbox_high)).all(dim=-1)
        log_p = log_prob_probability_flow(
            self.score_net,
            self.sde,
            theta,
            self.x_obs.to(self.device),
            hutchinson=True,
            n_eps=1,
        )
        return in_box & (log_p > self._kappa)

    # ------------------------------------------------------------------
    def sample(
        self, n: int, max_iters: int = 50, batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Rejection-sample the truncated proposal p̄^r(θ).

        Returns up to `n` accepted samples; raises if `max_iters` rounds of
        sampling did not yield enough accepted points.
        """
        accepted: list[torch.Tensor] = []
        n_accepted = 0
        bsz = batch_size or max(n, 1024)
        for _ in range(max_iters):
            cand = self.prior_sampler(bsz).to(self.device)
            mask = self.in_hpr(cand)
            if mask.any():
                accepted.append(cand[mask])
                n_accepted += int(mask.sum().item())
            if n_accepted >= n:
                break
        if not accepted:
            # Fall back to plain prior samples to keep training feasible.
            return self.prior_sampler(n).to(self.device)
        out = torch.cat(accepted, dim=0)[:n]
        if out.shape[0] < n:
            extra = self.prior_sampler(n - out.shape[0]).to(self.device)
            out = torch.cat([out, extra], dim=0)
        return out


def make_round_proposal(
    truncated_props: list[TruncatedProposal],
    prior_sampler: Callable[[int], torch.Tensor],
) -> Callable[[int], torch.Tensor]:
    """Return a sampler for the mixture p̃^r(θ) = (1/r) Σ_{s=0}^{r-1} p̄^s(θ).

    The s = 0 component is the prior (p̄^0 = p). For s >= 1 we use the
    TruncatedProposal objects already fitted in earlier rounds.
    """
    n_components = len(truncated_props) + 1  # +1 for the prior

    def sampler(n: int) -> torch.Tensor:
        per_component = [n // n_components] * n_components
        for k in range(n - sum(per_component)):
            per_component[k] += 1

        chunks: list[torch.Tensor] = []
        chunks.append(prior_sampler(per_component[0]))
        for s, prop in enumerate(truncated_props, start=1):
            if per_component[s] > 0:
                chunks.append(prop.sample(per_component[s]))
        return torch.cat(chunks, dim=0)

    return sampler
