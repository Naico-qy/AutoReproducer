"""
Gradient-based baselines used in Figure 5.1 of Cai et al. (2024):

* "Score"  : ADVI-style optimization but with the score-based divergence
             (eq. 2) as the objective instead of the negative ELBO.
* "Fisher" : same but with the standard (unweighted) Fisher divergence
             (Hyvarinen 2005).

Both are implemented with the reparameterization trick and Adam, matching
what the paper describes as "modified ADVI methods" used as additional
baselines in the Gaussian-target experiments of Section 5.1.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .bam import BaMState, _symmetrize
from .advi import Adam, _pack, _unpack
from .divergences import (
    score_based_divergence,
    fisher_divergence,
    gaussian_log_density_grad,
)


def _finite_diff_grad(
    fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """Central-difference gradient (slow but architecture-agnostic).

    Used as a JAX-free fallback so the codebase stays NumPy-only and
    compileable without further dependencies.  Suitable for small D.
    """
    g = np.zeros_like(theta)
    for i in range(theta.size):
        t_plus = theta.copy()
        t_minus = theta.copy()
        t_plus[i] += eps
        t_minus[i] -= eps
        g[i] = (fn(t_plus) - fn(t_minus)) / (2 * eps)
    return g


class _DivergenceVI:
    """Generic gradient-based optimizer that minimises a divergence."""

    def __init__(
        self,
        target_score_fn: Callable[[np.ndarray], np.ndarray],
        D: int,
        divergence: str,
        batch_size: int = 2,
        learning_rate: float = 1e-2,
        seed: int = 0,
    ) -> None:
        if divergence not in {"score", "fisher"}:
            raise ValueError(f"unknown divergence: {divergence}")
        self.target_score_fn = target_score_fn
        self.D = D
        self.divergence = divergence
        self.B = batch_size
        self.lr = learning_rate
        self.rng = np.random.default_rng(seed)
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    def _objective(self, theta: np.ndarray, eps: np.ndarray) -> float:
        D = self.D
        mu, L = _unpack(theta, D)
        Z = mu[None, :] + eps @ L.T
        Sigma = L @ L.T
        s = self.target_score_fn(Z)
        if self.divergence == "score":
            return score_based_divergence(Z, s, mu, Sigma)
        return fisher_divergence(Z, s, mu, Sigma)

    # ------------------------------------------------------------------
    def fit(
        self,
        mu0: np.ndarray,
        Sigma0: np.ndarray,
        n_iters: int,
        callback: Optional[Callable[[int, BaMState], None]] = None,
    ) -> BaMState:
        D = self.D
        L = np.linalg.cholesky(Sigma0 + 1e-10 * np.eye(D))
        theta = _pack(np.asarray(mu0, dtype=np.float64), L)
        opt = Adam(lr=self.lr)
        for t in range(n_iters):
            eps = self.rng.standard_normal(size=(self.B, D))

            def closure(th, eps=eps):
                return self._objective(th, eps)

            grad = _finite_diff_grad(closure, theta)
            theta = opt.step(theta, grad)
            mu, L = _unpack(theta, D)
            Sigma = _symmetrize(L @ L.T)
            self.history.append(
                {
                    "iter": t,
                    "n_grad_evals": (t + 1) * self.B,
                    "div": float(closure(theta)),
                }
            )
            if callback is not None:
                callback(t, BaMState(mu=mu, Sigma=Sigma))
        mu, L = _unpack(theta, D)
        return BaMState(mu=mu, Sigma=_symmetrize(L @ L.T))


class ScoreVI(_DivergenceVI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, divergence="score", **kwargs)


class FisherVI(_DivergenceVI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, divergence="fisher", **kwargs)
