"""
Gaussian Score Matching (GSM) baseline.

Reference (verified via citation-grounded retrieval)
------------------------------------------------------------
Modi, Margossian, Yao, Gower, Blei, Saul.
"Variational Inference with Gaussian Score Matching."
NeurIPS 2023. (Confirmed via DBLP entry returned by paper_search.)
URL: papers.nips.cc/paper_files/paper/2023/hash/5f9453c4848b89d4d8c5d6041f5fb9ec-Abstract-Conference.html

In the BaM paper, GSM is derived as the special limit of BaM with batch size
B = 1 and lambda -> infinity (Section 3.1 + Appendix C).  For B > 1, GSM
performs an ad hoc averaging across the per-sample updates as described in
Modi et al. (2023).  The single-sample update is the closed-form solution to

    mu_{t+1} = z   - Sigma_{t+1} g
    Sigma_{t+1} = exact match of one (z, g) pair

where (z, g) is the sample/score pair at the current iterate.

This module implements the per-sample analytic update along with the batch
averaging used in the original GSM paper.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .bam import BaMState, _symmetrize


def _gsm_single_update(state: BaMState, z: np.ndarray, g: np.ndarray) -> BaMState:
    """Exact single-sample GSM update.

    Solves for (mu_new, Sigma_new) such that the Gaussian variational density
    has score s_q(z) = -Sigma_new^{-1} (z - mu_new) equal to g, while
    minimizing KL(q_t || q_new).  The closed-form solution from
    Modi et al. (2023, Algorithm 1) is

        rho      = z - mu_t                 (D,)
        Sigma_g  = Sigma_t @ g              (D,)
        denom    = 1 + g^T (z - mu_t)
        Sigma_new = Sigma_t + (rho rho^T - Sigma_g Sigma_g^T) / denom
        mu_new    = z - Sigma_new @ g
    """
    mu_t, Sigma_t = state.mu, state.Sigma
    rho = z - mu_t
    Sg = Sigma_t @ g
    denom = 1.0 + float(g @ rho)
    if not np.isfinite(denom) or denom <= 1e-6:
        return state  # skip ill-conditioned step (denom must be > 0 for PD)
    Sigma_new = Sigma_t + (np.outer(rho, rho) - np.outer(Sg, Sg)) / denom
    Sigma_new = _symmetrize(Sigma_new)
    # Sanitize and project onto PD cone if a numerical glitch makes it indefinite.
    if not np.all(np.isfinite(Sigma_new)):
        return state
    w, V = np.linalg.eigh(Sigma_new)
    if not np.all(np.isfinite(w)):
        return state
    w = np.clip(w, a_min=1e-8, a_max=None)
    Sigma_new = (V * w) @ V.T
    mu_new = z - Sigma_new @ g
    return BaMState(mu=mu_new, Sigma=_symmetrize(Sigma_new))


class GSM:
    """Gaussian Score Matching VI driver."""

    def __init__(
        self,
        score_fn: Callable[[np.ndarray], np.ndarray],
        D: int,
        batch_size: int = 2,
        seed: int = 0,
    ) -> None:
        self.score_fn = score_fn
        self.D = D
        self.B = batch_size
        self.rng = np.random.default_rng(seed)
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    def _evaluate_scores(self, Z: np.ndarray) -> np.ndarray:
        try:
            return self.score_fn(Z)
        except (TypeError, ValueError):
            return np.stack([self.score_fn(z) for z in Z], axis=0)

    # ------------------------------------------------------------------
    def fit(
        self,
        mu0: np.ndarray,
        Sigma0: np.ndarray,
        n_iters: int,
        callback: Optional[Callable[[int, BaMState], None]] = None,
    ) -> BaMState:
        state = BaMState(
            mu=np.asarray(mu0, dtype=np.float64).copy(),
            Sigma=np.asarray(Sigma0, dtype=np.float64).copy(),
        )
        for t in range(n_iters):
            Z = state.sample(self.rng, self.B)
            G = self._evaluate_scores(Z)
            mus = []
            Sigmas = []
            for b in range(self.B):
                upd = _gsm_single_update(state, Z[b], G[b])
                mus.append(upd.mu)
                Sigmas.append(upd.Sigma)
            # Modi et al. (2023): average the candidate updates across the batch.
            mu_new = np.mean(np.stack(mus, axis=0), axis=0)
            Sigma_new = np.mean(np.stack(Sigmas, axis=0), axis=0)
            state = BaMState(mu=mu_new, Sigma=_symmetrize(Sigma_new))
            self.history.append(
                {
                    "iter": t,
                    "n_grad_evals": (t + 1) * self.B,
                    "mu_norm": float(np.linalg.norm(state.mu)),
                    "tr_Sigma": float(np.trace(state.Sigma)),
                }
            )
            if callback is not None:
                callback(t, state)
        return state
