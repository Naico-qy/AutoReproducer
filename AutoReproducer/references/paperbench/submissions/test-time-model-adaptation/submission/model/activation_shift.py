"""Back-to-Source Activation Shifting (Section 3.2 of FOA paper).

Implements Eqns. (7), (8) and (9):

    e_N^0  <-  e_N^0 + gamma * d_t                              (7)
    d_t    =  mu_N^S - mu_N(t)                                  (8)
    mu_N(t) = alpha * mu_N(X_t) + (1 - alpha) * mu_N(t-1)       (9)

with `alpha = 0.1` and `gamma = 1.0` (paper defaults).

Per addendum.md: "For the activation shifting moving average (Equation 9),
mu_N(0) should be initialized using the statistics of the first batch
mu_N(X_1)."
"""

from __future__ import annotations

from typing import Optional

import torch


class ActivationShifter:
    """Online EMA-based shift of the final-layer CLS feature toward the source
    in-distribution centroid.

    Parameters
    ----------
    mu_source : torch.Tensor of shape (D,)
        Source in-distribution mean of the final-layer CLS feature, computed
        once before TTA (paper Section 3.1, "Statistics calculation"; same set
        of samples reused for the shift centroid).
    gamma : float
        Step size in Eqn. (7).  Paper sets gamma = 1.0 ("aiming to exactly
        align the overall center of testing and training features").
    alpha : float
        EMA factor in Eqn. (9).  Paper sets alpha = 0.1.
    """

    def __init__(
        self,
        mu_source: torch.Tensor,
        gamma: float = 1.0,
        alpha: float = 0.1,
    ) -> None:
        if mu_source.dim() != 1:
            raise ValueError(
                f"mu_source must be a 1-D tensor of shape (D,), got {tuple(mu_source.shape)}"
            )
        self.mu_source = mu_source.detach().clone()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self._mu_running: Optional[torch.Tensor] = None  # mu_N(t)

    def reset(self) -> None:
        self._mu_running = None

    @torch.no_grad()
    def update_and_get_offset(self, batch_cls: torch.Tensor) -> torch.Tensor:
        """Update mu_N(t) with the new batch and return offset = gamma * d_t.

        Parameters
        ----------
        batch_cls : (B, D) tensor
            Per-sample N-th-layer CLS features of the current test batch X_t.

        Returns
        -------
        offset : (D,) tensor    -- gamma * (mu_N^S - mu_N(t))
        """
        mu_xt = batch_cls.mean(dim=0).detach()
        if self._mu_running is None:
            # Per addendum: initialize with the statistics of the first batch.
            self._mu_running = mu_xt.clone()
        else:
            self._mu_running = (
                self.alpha * mu_xt + (1.0 - self.alpha) * self._mu_running
            )
        d_t = self.mu_source.to(self._mu_running) - self._mu_running
        return self.gamma * d_t
