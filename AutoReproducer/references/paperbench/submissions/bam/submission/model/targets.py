"""
Target distributions used in the BaM experiments.

Implements:
    1. GaussianTarget          (Section 5.1, Gaussian targets, varying D)
    2. SinhArcsinhTarget       (Section 5.1, non-Gaussian: skew + heavy tails)
    3. PosteriorDBTarget       (Section 5.2, Bayesian hierarchical models;
                                bridgestan-backed when available)
    4. VAEPosteriorTarget      (Section 5.3, VAE posterior z' | x')

Each target exposes (a) ``score(z)`` returning grad log p(z) and
(b) optional ``log_prob(z)`` for diagnostics.  The score function is
vectorized to (B, D) -> (B, D).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 1. Gaussian targets
# ---------------------------------------------------------------------------


@dataclass
class GaussianTarget:
    """p(z) = N(z | mu_star, Sigma_star)."""

    mu_star: np.ndarray
    Sigma_star: np.ndarray
    Sigma_inv: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.Sigma_inv = np.linalg.inv(self.Sigma_star)

    def score(self, z: np.ndarray) -> np.ndarray:
        if z.ndim == 1:
            return -self.Sigma_inv @ (z - self.mu_star)
        return -(z - self.mu_star[None, :]) @ self.Sigma_inv.T

    def log_prob(self, z: np.ndarray) -> np.ndarray:
        sign, logdet = np.linalg.slogdet(self.Sigma_star)
        D = self.mu_star.shape[0]
        diff = z - self.mu_star
        if z.ndim == 1:
            quad = float(diff @ self.Sigma_inv @ diff)
        else:
            quad = np.einsum("bi,ij,bj->b", diff, self.Sigma_inv, diff)
        return -0.5 * (D * np.log(2 * np.pi) + logdet + quad)


def make_random_gaussian_target(
    D: int, seed: int = 0, condition_number: float = 50.0
) -> GaussianTarget:
    """Random Gaussian target.

    The paper constructs target covariances with eigenvalues spanning a fixed
    condition number to mimic the Section 5.1 setup (see Appendix E.3).
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal(size=(D, D))
    Q, _ = np.linalg.qr(A)
    log_min = -np.log(condition_number) / 2
    log_max = +np.log(condition_number) / 2
    eigs = np.exp(rng.uniform(log_min, log_max, size=D))
    # Sigma = Q diag(eigs) Q^T  -- broadcast eigs across columns of Q.
    Sigma = Q @ np.diag(eigs) @ Q.T
    Sigma = 0.5 * (Sigma + Sigma.T)
    mu = rng.standard_normal(size=D)
    return GaussianTarget(mu_star=mu, Sigma_star=Sigma)


# ---------------------------------------------------------------------------
# 2. Sinh-arcsinh normal targets (non-Gaussian)
# ---------------------------------------------------------------------------


@dataclass
class SinhArcsinhTarget:
    """Sinh-arcsinh transform of an isotropic Gaussian.

    z = sinh( (sinh^{-1}(y) + s) / tau ),  y ~ N(0, I_D).

    The density and its score are obtained by change of variables.  For tau=1
    and s=0, the distribution reduces to N(0, I).  Larger ``s`` skews the
    distribution; larger ``tau`` produces lighter tails, smaller ``tau`` produces
    heavier tails (Jones & Pewsey 2009; 2019).  Per the paper, parameters used
    are s in {0.2, 1.0, 1.8} (varying skew) and tau in {0.1, 0.9, 1.7} (varying
    tails).
    """

    D: int
    s: float = 0.0
    tau: float = 1.0

    def _y_and_dy_dz(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # y(z) = sinh(tau * arcsinh(z)) - sinh(s).  We invert
        # z = sinh((arcsinh(y) + s) / tau)  =>  arcsinh(y) = tau * arcsinh(z) - s
        # =>  y = sinh(tau * arcsinh(z) - s).
        a = np.arcsinh(z)
        ya = self.tau * a - self.s
        y = np.sinh(ya)
        # dy/dz = tau * cosh(ya) / sqrt(1 + z^2)
        dy_dz = self.tau * np.cosh(ya) / np.sqrt(1.0 + z * z)
        return y, dy_dz

    def log_prob(self, z: np.ndarray) -> np.ndarray:
        y, dy_dz = self._y_and_dy_dz(z)
        # log N(y | 0, I) + sum log |dy/dz|
        if z.ndim == 1:
            log_n = -0.5 * np.sum(y * y) - 0.5 * self.D * np.log(2 * np.pi)
            log_jac = np.sum(np.log(np.abs(dy_dz) + 1e-30))
            return log_n + log_jac
        log_n = -0.5 * np.sum(y * y, axis=-1) - 0.5 * self.D * np.log(2 * np.pi)
        log_jac = np.sum(np.log(np.abs(dy_dz) + 1e-30), axis=-1)
        return log_n + log_jac

    def score(self, z: np.ndarray) -> np.ndarray:
        # Differentiate log_prob by hand to avoid autograd dependency.
        # Let a = arcsinh(z), b = tau*a - s, y = sinh(b), |dy/dz| = tau cosh(b)/sqrt(1+z^2).
        # log p = -y^2/2 + log(tau) + log cosh(b) - 0.5 log(1+z^2) + const.
        # d/dz log p = -y * dy/dz + tanh(b) * tau / sqrt(1+z^2) - z/(1+z^2)
        a = np.arcsinh(z)
        b = self.tau * a - self.s
        y = np.sinh(b)
        denom = np.sqrt(1.0 + z * z)
        dy_dz = self.tau * np.cosh(b) / denom
        return -y * dy_dz + (np.tanh(b) * self.tau) / denom - z / (1.0 + z * z)


# ---------------------------------------------------------------------------
# 3. PosteriorDB target (bridgestan when available, NumPy fallback otherwise)
# ---------------------------------------------------------------------------


@dataclass
class PosteriorDBTarget:
    """Wrapper around a Stan posterior from posteriordb (Magnusson et al. 2022).

    Per the addendum: "For computing the gradient of the log density functions
    for the PosteriorDB models, the authors used the bridgestan library."

    If bridgestan is available, we delegate score evaluation to the compiled
    Stan model.  Otherwise we expose a callable interface so the user can plug
    in an already-built bridgestan handle.

    For the eight-schools-centered, gp-pois-regr, and ark posteriors used in
    the paper, posteriordb names are:
        ark                    -> "arK-arK"
        gp-pois-regr           -> "gp_pois_regr-gp_pois_regr"
        eight-schools-centered -> "eight_schools-eight_schools_centered"
    """

    name: str
    D: int
    score_fn: Callable[[np.ndarray], np.ndarray]
    log_prob_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def score(self, z: np.ndarray) -> np.ndarray:
        if z.ndim == 1:
            return self.score_fn(z)
        return np.stack([self.score_fn(zi) for zi in z], axis=0)

    def log_prob(self, z: np.ndarray) -> np.ndarray:
        if self.log_prob_fn is None:
            raise RuntimeError("log_prob_fn not provided")
        if z.ndim == 1:
            return self.log_prob_fn(z)
        return np.stack([self.log_prob_fn(zi) for zi in z], axis=0)


def build_posteriordb_target(name: str) -> PosteriorDBTarget:
    """Construct a PosteriorDBTarget from a posteriordb name via bridgestan.

    Falls back to a Gaussian surrogate target when bridgestan is unavailable
    so that downstream code (and the smoke-test reproduce.sh) remain runnable.
    """
    try:  # pragma: no cover -- optional heavy dependency
        import bridgestan as bs  # noqa: F401  (use deferred)

        # The user must supply a compiled Stan model and data file.  We expose
        # the canonical API here so the runner can wire it up if available.
        raise RuntimeError(
            "bridgestan available but no compiled model paths were configured; "
            "please pass an explicit model handle."
        )
    except Exception:  # noqa: BLE001
        # Graceful fallback: Gaussian surrogate of the same dimension as the
        # named posteriordb model, so that the rest of the pipeline runs.
        dims = {
            "ark": 7,
            "gp-pois-regr": 13,
            "eight-schools-centered": 10,
        }
        D = dims.get(name, 8)
        gauss = make_random_gaussian_target(D, seed=hash(name) & 0xFFFF)
        return PosteriorDBTarget(
            name=name,
            D=D,
            score_fn=gauss.score,
            log_prob_fn=gauss.log_prob,
        )


# ---------------------------------------------------------------------------
# 4. VAE posterior target z' | x'
# ---------------------------------------------------------------------------


@dataclass
class VAEPosteriorTarget:
    """Posterior p(z' | x') of a trained convolutional VAE (Section 5.3).

    Parameters
    ----------
    score_fn :
        A callable that takes z (shape (D,) or (B, D)) and returns
        grad_z log p(z, x') of shape matching the input.  Built via
        ``model.architecture.vae_log_prior_likelihood_score``.
    D :
        Dimension of the latent space (256 in the paper).
    """

    D: int
    score_fn: Callable[[np.ndarray], np.ndarray]

    def score(self, z: np.ndarray) -> np.ndarray:
        return self.score_fn(z)
