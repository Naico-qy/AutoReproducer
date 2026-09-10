"""
Divergence and metric utilities used by BaM and the baselines.

The score-based divergence (eq. 2 of Cai et al. 2024) is

    D(q; p) = E_q [ || nabla log (q/p) ||^2_{Cov(q)} ].

Its plug-in Monte-Carlo estimator (eq. 3) uses ``B`` samples from ``q``:

    D_hat(q; p) = (1/B) sum_b ||grad log q(z_b) - grad log p(z_b)||^2_{Cov(q)}.

For Gaussian ``q = N(mu, Sigma)``, grad log q(z) = -Sigma^{-1} (z - mu).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Variational gradient
# ---------------------------------------------------------------------------


def gaussian_log_density_grad(
    z: np.ndarray, mu: np.ndarray, Sigma_inv: np.ndarray
) -> np.ndarray:
    """grad_z log N(z | mu, Sigma) = - Sigma^{-1} (z - mu).  Vectorized."""
    if z.ndim == 1:
        return -Sigma_inv @ (z - mu)
    return -(z - mu[None, :]) @ Sigma_inv.T


# ---------------------------------------------------------------------------
# Score-based and Fisher divergences
# ---------------------------------------------------------------------------


def score_based_divergence(
    Z: np.ndarray, target_score: np.ndarray, mu: np.ndarray, Sigma: np.ndarray
) -> float:
    """Plug-in Monte-Carlo estimate of D(q; p) — eq. (3) of Cai et al. 2024."""
    Sigma_inv = np.linalg.inv(Sigma)
    q_score = gaussian_log_density_grad(Z, mu, Sigma_inv)
    diff = q_score - target_score  # nabla log q - nabla log p
    # Sigma-weighted norm: ||x||^2_{Cov(q)} = x^T Sigma x.
    weighted = np.einsum("bi,ij,bj->b", diff, Sigma, diff)
    return float(np.mean(weighted))


def fisher_divergence(
    Z: np.ndarray, target_score: np.ndarray, mu: np.ndarray, Sigma: np.ndarray
) -> float:
    """Plug-in Monte-Carlo estimate of E_q ||grad log q/p||^2 (Fisher divergence)."""
    Sigma_inv = np.linalg.inv(Sigma)
    q_score = gaussian_log_density_grad(Z, mu, Sigma_inv)
    diff = q_score - target_score
    return float(np.mean(np.sum(diff * diff, axis=-1)))


# ---------------------------------------------------------------------------
# KL divergences (used as evaluation metrics on Gaussian targets)
# ---------------------------------------------------------------------------


def reverse_kl_gaussian(
    mu_q: np.ndarray, Sigma_q: np.ndarray, mu_p: np.ndarray, Sigma_p: np.ndarray
) -> float:
    """KL(q || p) for two Gaussians (closed form)."""
    D = mu_q.shape[0]
    Sigma_p_inv = np.linalg.inv(Sigma_p)
    diff = mu_p - mu_q
    sign_q, logdet_q = np.linalg.slogdet(Sigma_q)
    sign_p, logdet_p = np.linalg.slogdet(Sigma_p)
    return 0.5 * float(
        np.trace(Sigma_p_inv @ Sigma_q)
        + diff @ Sigma_p_inv @ diff
        - D
        + (logdet_p - logdet_q)
    )


def forward_kl_gaussian(
    mu_q: np.ndarray, Sigma_q: np.ndarray, mu_p: np.ndarray, Sigma_p: np.ndarray
) -> float:
    """KL(p || q) for two Gaussians (closed form)."""
    return reverse_kl_gaussian(mu_p, Sigma_p, mu_q, Sigma_q)


def relative_mean_error(mu: np.ndarray, mu_ref: np.ndarray) -> float:
    """L2 relative error in mean (used in Section 5.2)."""
    return float(np.linalg.norm(mu - mu_ref) / max(np.linalg.norm(mu_ref), 1e-12))


def relative_sd_error(Sigma: np.ndarray, Sigma_ref: np.ndarray) -> float:
    """L2 relative error in posterior std (used in Section 5.2)."""
    sd = np.sqrt(np.clip(np.diag(Sigma), 1e-30, None))
    sd_ref = np.sqrt(np.clip(np.diag(Sigma_ref), 1e-30, None))
    return float(np.linalg.norm(sd - sd_ref) / max(np.linalg.norm(sd_ref), 1e-12))


# ---------------------------------------------------------------------------
# Monte-Carlo wrappers (when target is non-Gaussian, we need MC samples)
# ---------------------------------------------------------------------------


def monte_carlo_score_div(
    target_score_fn: Callable[[np.ndarray], np.ndarray],
    mu: np.ndarray,
    Sigma: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = 256,
) -> float:
    L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(mu.shape[0]))
    eps = rng.standard_normal(size=(n_samples, mu.shape[0]))
    Z = mu[None, :] + eps @ L.T
    s = target_score_fn(Z)
    return score_based_divergence(Z, s, mu, Sigma)
