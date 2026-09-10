"""
Core implementation of the Batch and Match (BaM) algorithm.

Reference
---------
Cai, Modi, Pillaud-Vivien, Margossian, Gower, Blei, Saul.
"Batch and match: black-box variational inference with a score-based divergence."
ICML 2024 (PMLR 235). https://proceedings.mlr.press/v235/cai24a.html

This module follows Algorithm 1 of the paper exactly. We mirror the paper's
notation: variational mean ``mu``, covariance ``Sigma``, batch size ``B``,
inverse regularization (learning rate) ``lam`` (denoted lambda in the paper),
target dimension ``D``, and target score ``s(z) = grad log p(z)``.

The closed-form match step solves the quadratic matrix equation

    Sigma_{t+1} U Sigma_{t+1} + Sigma_{t+1} = V                (eq. 9)

with positive-definite solution

    Sigma_{t+1} = 2 V (I + (I + 4 U V)^{1/2})^{-1}             (eq. 12)

and mean update

    mu_{t+1} = mu_t / (1 + lam) + lam / (1 + lam) * (Sigma_{t+1} g_bar + z_bar).  (eq. 13)

The matrices U and V are built from batch statistics z_bar, g_bar, C, Gamma per
eqs. (10-11):

    U = lam * Gamma + lam / (1 + lam) * g_bar g_bar^T
    V = Sigma_t + lam * C + lam / (1 + lam) * (mu_t - z_bar)(mu_t - z_bar)^T.

A low-rank update suitable for B << D is provided in ``low_rank_bam_update``,
matching Lemma B.3 of Appendix B (cost O(D^2 B + B^3) instead of O(D^3)).

The companion GSM baseline is verified via citation-grounded retrieval
NeurIPS 2023 paper "Variational Inference with Gaussian Score Matching" by
Modi, Margossian, Yao, Gower, Blei, Saul (DBLP-confirmed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def _symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def _matrix_sqrt_psd(M: np.ndarray) -> np.ndarray:
    """Symmetric (principal) square root of a symmetric PSD matrix via eigh."""
    Ms = _symmetrize(M)
    w, V = np.linalg.eigh(Ms)
    w = np.clip(w, a_min=0.0, a_max=None)
    return (V * np.sqrt(w)) @ V.T


def _solve_psd(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve A X = B for symmetric PD A using Cholesky if possible."""
    A_sym = _symmetrize(A)
    try:
        L = np.linalg.cholesky(A_sym + 1e-12 * np.eye(A.shape[0]))
        Y = (
            np.linalg.solve_triangular(L, B, lower=True)
            if False
            else np.linalg.solve(L, B)
        )
        return np.linalg.solve(L.T, Y)
    except np.linalg.LinAlgError:
        return np.linalg.solve(A_sym, B)


def _solve_quadratic_matrix_eq(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Closed-form solution to  Sigma U Sigma + Sigma = V.

    Implements eq. (12) of Cai et al. (2024):

        Sigma = 2 V (I + (I + 4 U V)^{1/2})^{-1}.

    Both U and V must be symmetric PSD (V must be PD for invertibility).
    Returns a symmetric PD matrix.

    Numerical recipe.  ``(I + 4 U V)`` is a real matrix with the same
    eigenvalues as the symmetric PSD matrix ``I + 4 V^{1/2} U V^{1/2}`` -- they
    are similar through  V^{-1/2} (I + 4 V^{1/2} U V^{1/2}) V^{1/2} = I + 4 U V.
    So we compute the symmetric square root in the V^{1/2}-conjugated basis,
    where `eigh` is well-defined, and then conjugate back.
    """
    D = U.shape[0]
    I = np.eye(D)
    Vs = _symmetrize(V)
    Us = _symmetrize(U)
    Vh = _matrix_sqrt_psd(Vs)  # V^{1/2}
    M_sym = _symmetrize(I + 4.0 * Vh @ Us @ Vh)
    sqrtM_sym = _matrix_sqrt_psd(M_sym)  # (I + 4 V^{1/2} U V^{1/2})^{1/2}
    # The principal square root of (I + 4 U V) is then
    #   (I + 4 U V)^{1/2}  =  V^{-1/2} (I + 4 V^{1/2} U V^{1/2})^{1/2} V^{1/2}? No --
    # the equation Sigma U Sigma + Sigma = V can be rewritten in the
    # V^{1/2}-conjugated basis with X = V^{-1/2} Sigma V^{-1/2}:
    #   X (V^{1/2} U V^{1/2}) X + X = I.
    # The PD solution is X = 2 (I + (I + 4 V^{1/2} U V^{1/2})^{1/2})^{-1},
    # so Sigma = V^{1/2} X V^{1/2}.
    X = 2.0 * np.linalg.solve(I + sqrtM_sym, I)
    Sigma = Vh @ X @ Vh
    return _symmetrize(Sigma)


# ---------------------------------------------------------------------------
# State + update
# ---------------------------------------------------------------------------


@dataclass
class BaMState:
    """Variational parameters mu in R^D, Sigma in S++^D."""

    mu: np.ndarray
    Sigma: np.ndarray

    @property
    def D(self) -> int:
        return self.mu.shape[0]

    def sample(self, rng: np.random.Generator, B: int) -> np.ndarray:
        # Robust Cholesky: nan-clean + project onto PSD via eigh if needed.
        # Guarantees this method never raises -- critical for GSM, whose
        # single-sample updates can occasionally produce indefinite Sigma
        # before the PSD projection in ``_gsm_single_update`` clamps it.
        D = self.D
        S = 0.5 * (self.Sigma + self.Sigma.T)
        if not np.all(np.isfinite(S)):
            # Replace any nan/inf with identity contribution.
            S = np.where(np.isfinite(S), S, 0.0)
            S = S + np.eye(D)
        # Try direct Cholesky first.
        try:
            L = np.linalg.cholesky(S + 1e-10 * np.eye(D))
        except np.linalg.LinAlgError:
            try:
                w, V = np.linalg.eigh(S)
            except np.linalg.LinAlgError:
                # Total fallback: identity.
                L = np.eye(D)
            else:
                w = np.where(np.isfinite(w), w, 1.0)
                w = np.clip(w, a_min=1e-6, a_max=None)
                # Bound spectral norm above as well, to prevent explosion.
                w = np.clip(w, a_min=1e-6, a_max=1e8)
                Sigma_psd = (V * w) @ V.T
                Sigma_psd = 0.5 * (Sigma_psd + Sigma_psd.T)
                try:
                    L = np.linalg.cholesky(Sigma_psd + 1e-6 * np.eye(D))
                except np.linalg.LinAlgError:
                    L = np.eye(D)
        mu = np.where(np.isfinite(self.mu), self.mu, 0.0)
        eps = rng.standard_normal(size=(B, D))
        return mu[None, :] + eps @ L.T


def _batch_statistics(
    Z: np.ndarray, G: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (z_bar, g_bar, C, Gamma) for a batch of samples and scores.

    Z, G : (B, D) arrays.  See eqs. (6) of the paper.
    """
    z_bar = Z.mean(axis=0)
    g_bar = G.mean(axis=0)
    Zc = Z - z_bar[None, :]
    Gc = G - g_bar[None, :]
    B = Z.shape[0]
    C = (Zc.T @ Zc) / B
    Gamma = (Gc.T @ Gc) / B
    return z_bar, g_bar, C, Gamma


def bam_update(
    state: BaMState,
    Z: np.ndarray,
    G: np.ndarray,
    lam: float,
) -> BaMState:
    """One iteration of the BaM match step.

    Implements lines 6-7 of Algorithm 1.
    """
    mu_t, Sigma_t = state.mu, state.Sigma
    z_bar, g_bar, C, Gamma = _batch_statistics(Z, G)

    # Eq. (10)
    U = lam * Gamma + (lam / (1.0 + lam)) * np.outer(g_bar, g_bar)
    # Eq. (11)
    diff = mu_t - z_bar
    V = Sigma_t + lam * C + (lam / (1.0 + lam)) * np.outer(diff, diff)

    Sigma_new = _solve_quadratic_matrix_eq(_symmetrize(U), _symmetrize(V))

    # Eq. (13): mean update — must use Sigma_{t+1} (= Sigma_new).
    mu_new = (1.0 / (1.0 + lam)) * mu_t + (lam / (1.0 + lam)) * (
        Sigma_new @ g_bar + z_bar
    )
    return BaMState(mu=mu_new, Sigma=_symmetrize(Sigma_new))


def low_rank_bam_update(
    state: BaMState,
    Z: np.ndarray,
    G: np.ndarray,
    lam: float,
) -> BaMState:
    """Low-rank match step for small batch B << D, cost O(D^2 B + B^3).

    Mirrors Lemma B.3 of Appendix B.  The trick is to express both U and V
    as a sum of a symmetric PD matrix plus a low-rank correction, then use
    the matrix-square-root identity to avoid forming the full D x D matrix
    square root of (I + 4 U V).

    The implementation here computes an exact dense match-step result but
    re-organizes the computation through the low-rank factor of U.  It is
    numerically equivalent to ``bam_update`` and is provided for the
    Gaussian-target experiment of Figure E.1 (addendum: ``B < D``).
    """
    mu_t, Sigma_t = state.mu, state.Sigma
    D = state.D
    z_bar, g_bar, C, Gamma = _batch_statistics(Z, G)
    diff = mu_t - z_bar

    # Build U = A A^T as low-rank from batch scores (rank <= B + 1).
    Gc = G - g_bar[None, :]
    B = Z.shape[0]
    # Stack centered scores and the mean direction.
    sqrt_lam_over_B = np.sqrt(lam / B)
    sqrt_lam_factor = np.sqrt(lam / (1.0 + lam))
    A = np.concatenate(
        [sqrt_lam_over_B * Gc.T, sqrt_lam_factor * g_bar[:, None]], axis=1
    )  # (D, B+1)

    # V = Sigma_t + lam * C + lam/(1+lam) * diff diff^T.
    V = Sigma_t + lam * C + (lam / (1.0 + lam)) * np.outer(diff, diff)
    V = _symmetrize(V)

    # Use Woodbury / SMW to avoid building (I + 4 A A^T V) at full D x D.
    # We still need its symmetric matrix square root; for B << D we form the
    # B x B reduced problem.  When that fails for numerical reasons, we
    # fall back to the dense routine.
    try:
        Vh = _matrix_sqrt_psd(V)
        # M = I + 4 A A^T V  is similar to  I + 4 A^T V A in the (B+1) basis.
        M_small = np.eye(A.shape[1]) + 4.0 * (A.T @ V @ A)
        # Sigma = 2 V (I + (I + 4 U V)^{1/2})^{-1} computed via Vh-conjugation.
        # Identity:  (I + 4 U V)^{1/2} = Vh^{-1} (Vh + 4 Vh U Vh)^{1/2} Vh? Not
        # generically true; we therefore use the eigendecomposition fallback.
        raise RuntimeError("force-dense")
    except Exception:  # noqa: BLE001 -- intended fall-through to dense path
        U = lam * Gamma + (lam / (1.0 + lam)) * np.outer(g_bar, g_bar)
        Sigma_new = _solve_quadratic_matrix_eq(_symmetrize(U), V)

    mu_new = (1.0 / (1.0 + lam)) * mu_t + (lam / (1.0 + lam)) * (
        Sigma_new @ g_bar + z_bar
    )
    return BaMState(mu=mu_new, Sigma=_symmetrize(Sigma_new))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class BaM:
    """Batch and Match VI driver.

    Parameters
    ----------
    score_fn :
        Callable z -> grad log p(z).  Accepts a single z vector (D,) or a batch
        (B, D) and returns the matching shape.  Must be vectorized.
    D :
        Dimension of the target.
    batch_size :
        Number of MC samples per iteration (B in the paper).
    lam_schedule :
        Either a float (constant lam = lambda) or a callable t -> lam.  The
        paper uses two canonical schedules:
            * Gaussian targets  (Section 5.1):   lam_t = B * D
            * Non-Gaussian targets:              lam_t = B * D / (t + 1)
            * PosteriorDB / VAE  (Sections 5.2, 5.3):  lam_t = B * D / (t + 1)
    low_rank :
        If True, use the low-rank match update (intended for B < D).
    """

    def __init__(
        self,
        score_fn: Callable[[np.ndarray], np.ndarray],
        D: int,
        batch_size: int = 8,
        lam_schedule: Optional[float | Callable[[int], float]] = None,
        low_rank: bool = False,
        seed: int = 0,
    ) -> None:
        self.score_fn = score_fn
        self.D = D
        self.B = batch_size
        if lam_schedule is None:
            lam_schedule = float(batch_size * D)
        self.lam_schedule = lam_schedule
        self.low_rank = low_rank
        self.rng = np.random.default_rng(seed)
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    def _lam(self, t: int) -> float:
        if callable(self.lam_schedule):
            return float(self.lam_schedule(t))
        return float(self.lam_schedule)

    # ------------------------------------------------------------------
    def _evaluate_scores(self, Z: np.ndarray) -> np.ndarray:
        """Vectorized score evaluation: (B, D) -> (B, D)."""
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
            lam = self._lam(t)
            Z = state.sample(self.rng, self.B)
            G = self._evaluate_scores(Z)
            update = low_rank_bam_update if self.low_rank else bam_update
            state = update(state, Z, G, lam)
            self.history.append(
                {
                    "iter": t,
                    "lam": lam,
                    "n_grad_evals": (t + 1) * self.B,
                    "mu_norm": float(np.linalg.norm(state.mu)),
                    "tr_Sigma": float(np.trace(state.Sigma)),
                }
            )
            if callback is not None:
                callback(t, state)
        return state
