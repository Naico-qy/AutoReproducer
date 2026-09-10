"""
Automatic Differentiation Variational Inference (ADVI) baseline.

Reference
---------
Kucukelbir, Tran, Ranganath, Gelman, Blei.
"Automatic Differentiation Variational Inference."
Journal of Machine Learning Research 18(14):1-45, 2017.

The BaM paper (Section 5) compares against a Gaussian-full-covariance ADVI
implementation that maximises the ELBO via SGD using the reparameterization
trick (eq. 7 of Cai et al. 2024).  We implement that here in NumPy with a
Cholesky parameterization of Sigma:

    Sigma = L L^T,   L = lower-triangular with positive diagonal.

The unconstrained parameter vector contains (mu, off-diagonal entries of L,
log-diagonal of L).  The reparameterized sample is

    z = mu + L * eps,   eps ~ N(0, I).

We use Adam (Kingma & Ba 2015) with a learning rate selected via grid search,
following the BaM paper's evaluation protocol.  The negative ELBO is

    -ELBO = -E_q[log p(z)] - H[q]   with  H[q] = sum log diag(L) + const.

Following the addendum, the deep-generative-model experiment uses one MC sample
per gradient (mc_sim = 1).  All other experiments support the same setting and
default to it.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .bam import BaMState, _symmetrize


# ---------------------------------------------------------------------------
# Adam optimiser
# ---------------------------------------------------------------------------


class Adam:
    """Minimal Adam (Kingma & Ba 2015)."""

    def __init__(
        self, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.t = 0

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.m is None or self.v is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)
        self.t += 1
        m = self.beta1 * self.m + (1 - self.beta1) * grad
        v = self.beta2 * self.v + (1 - self.beta2) * grad * grad
        self.m = m
        self.v = v
        m_hat = m / (1 - self.beta1**self.t)
        v_hat = v / (1 - self.beta2**self.t)
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# (un)packing of variational parameters
# ---------------------------------------------------------------------------


def _pack(mu: np.ndarray, L: np.ndarray) -> np.ndarray:
    D = mu.shape[0]
    log_diag = np.log(np.clip(np.diag(L), a_min=1e-12, a_max=None))
    tril_idx = np.tril_indices(D, k=-1)
    off_diag = L[tril_idx]
    return np.concatenate([mu, log_diag, off_diag], axis=0)


def _unpack(theta: np.ndarray, D: int) -> tuple[np.ndarray, np.ndarray]:
    mu = theta[:D]
    log_diag = theta[D : 2 * D]
    off_diag = theta[2 * D :]
    L = np.zeros((D, D))
    np.fill_diagonal(L, np.exp(log_diag))
    tril_idx = np.tril_indices(D, k=-1)
    L[tril_idx] = off_diag
    return mu, L


# ---------------------------------------------------------------------------
# ELBO and gradients
# ---------------------------------------------------------------------------


def _neg_elbo_and_grad(
    theta: np.ndarray,
    D: int,
    log_p_and_score: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    mc_sim: int = 1,
) -> tuple[float, np.ndarray]:
    """Negative ELBO with reparameterized gradients.

    log_p_and_score(z) -> (log p(z), grad log p(z)) for a batch (B, D).
    """
    mu, L = _unpack(theta, D)
    eps = rng.standard_normal(size=(mc_sim, D))
    Z = mu[None, :] + eps @ L.T  # reparam samples
    log_p, score = log_p_and_score(Z)
    # Grad wrt mu: average score.
    grad_mu = -score.mean(axis=0)
    # Grad wrt L from log p:  score^T eps (lower triangular contribution).
    # d z / d L = eps in transposed sense; chain rule:
    #   d log p / d L_{ij} = sum_b score_b[i] * eps_b[j]   (i >= j)
    #   plus entropy term: H = sum_i log L_{ii}, so dH/dL_{ii} = 1/L_{ii}.
    grad_L_full = (-score[:, :, None] * eps[:, None, :]).mean(axis=0)
    tril_mask = np.tri(D, k=0, dtype=bool)
    grad_L = np.where(tril_mask, grad_L_full, 0.0)
    # Entropy gradient: -log |L| component contributes -1/L_ii on diagonal of -ELBO.
    diag_idx = np.arange(D)
    grad_L[diag_idx, diag_idx] -= 1.0 / np.clip(np.diag(L), 1e-12, None)
    # Repack: log_diag uses chain rule d/d log_diag = L_ii * d/dL_ii.
    grad_log_diag = grad_L[diag_idx, diag_idx] * np.diag(L)
    tril_idx = np.tril_indices(D, k=-1)
    grad_off = grad_L[tril_idx]
    grad = np.concatenate([grad_mu, grad_log_diag, grad_off], axis=0)
    neg_elbo = float(-log_p.mean() - np.sum(np.log(np.clip(np.diag(L), 1e-12, None))))
    return neg_elbo, grad


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class ADVI:
    """Full-covariance Gaussian ADVI with Adam."""

    def __init__(
        self,
        log_p_and_score: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
        D: int,
        batch_size: int = 1,
        learning_rate: float = 1e-2,
        seed: int = 0,
    ) -> None:
        self.log_p_and_score = log_p_and_score
        self.D = D
        self.B = batch_size
        self.lr = learning_rate
        self.rng = np.random.default_rng(seed)
        self.history: list[dict] = []

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
            neg_elbo, grad = _neg_elbo_and_grad(
                theta, D, self.log_p_and_score, self.rng, mc_sim=self.B
            )
            theta = opt.step(theta, grad)
            mu, L = _unpack(theta, D)
            Sigma = _symmetrize(L @ L.T)
            self.history.append(
                {
                    "iter": t,
                    "n_grad_evals": (t + 1) * self.B,
                    "neg_elbo": neg_elbo,
                    "mu_norm": float(np.linalg.norm(mu)),
                    "tr_Sigma": float(np.trace(Sigma)),
                }
            )
            if callback is not None:
                callback(t, BaMState(mu=mu, Sigma=Sigma))
        mu, L = _unpack(theta, D)
        return BaMState(mu=mu, Sigma=_symmetrize(L @ L.T))
