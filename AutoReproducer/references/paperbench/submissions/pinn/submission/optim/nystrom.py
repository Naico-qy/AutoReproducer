"""Randomized Nyström approximation (Algorithm 5, Appendix E.2).

We follow Frangella, Tropp, Udell (2023) "Randomized Nyström
Preconditioning" (arXiv:2110.02820 / SIAM J. Matrix Anal. Appl. 44(2):
718–752, 2023) — the same construction used in NNCG of Rathore et al.
(2024).

The algorithm computes a top-s approximate eigendecomposition of a
symmetric (positive-semidefinite) matrix `A`:

    [U, Λ̂] = RandomizedNyströmApproximation(A, s)

with the `chol` fallback for indefinite Hessians (the red portion of
Algorithm 5 in the paper).

The matrix A is accessed only through Hessian-vector products, exposed
via a callable hvp(v) -> torch.Tensor in this module.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import torch


class NystromPreconditioner(NamedTuple):
    U: torch.Tensor  # (p, s)
    eigs: torch.Tensor  # (s,)
    sketch_size: int


def _hvp_matrix(
    hvp: Callable[[torch.Tensor], torch.Tensor], Q: torch.Tensor
) -> torch.Tensor:
    """Apply hvp to each column of Q, returning A Q.

    Q has shape (p, s); we evaluate column-by-column. This matches the
    paper's note that the sketch Y = M Q is implemented via Hessian-
    vector products (Appendix E.2).
    """
    cols = []
    for j in range(Q.shape[1]):
        cols.append(hvp(Q[:, j]).detach())
    return torch.stack(cols, dim=1)


def randomized_nystrom_approximation(
    hvp: Callable[[torch.Tensor], torch.Tensor],
    p: int,
    s: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> NystromPreconditioner:
    """Algorithm 5 (RandomizedNyströmApproximation) from Appendix E.2.

    Parameters
    ----------
    hvp : callable v -> H v
        Hessian-vector product oracle.
    p : int
        Ambient dimension (number of model parameters).
    s : int
        Sketch size.  In the paper s = 60.
    """
    device = device or torch.device("cpu")
    if generator is None:
        S = torch.randn(p, s, device=device, dtype=dtype)
    else:
        S = torch.randn(p, s, device=device, dtype=dtype, generator=generator)
    # QR with reduced (economy) decomposition.
    Q, _ = torch.linalg.qr(S, mode="reduced")  # (p, s)

    Y = _hvp_matrix(hvp, Q)  # (p, s)

    # Shift for numerical stability: ν = sqrt(p) eps(||Y||_2)
    Y_norm = torch.linalg.norm(Y, ord=2)
    eps_y = torch.finfo(dtype).eps * (Y_norm + 1e-30)
    nu = torch.sqrt(torch.tensor(float(p), device=device, dtype=dtype)) * eps_y
    Y_nu = Y + nu * Q

    QtY = Q.T @ Y_nu  # (s, s)
    QtY = 0.5 * (QtY + QtY.T)  # symmetrize

    lam_shift = torch.zeros((), device=device, dtype=dtype)
    try:
        C = torch.linalg.cholesky(QtY)
        B = torch.linalg.solve_triangular(C, Y_nu.T, upper=False).T
    except RuntimeError:
        # Fail-safe: indefinite QtY (red portion of Algorithm 5).
        eigvals, eigvecs = torch.linalg.eigh(QtY)
        lam_shift = -eigvals.min().clamp(
            max=torch.zeros((), device=device, dtype=dtype)
        )
        QtY_pd = QtY + lam_shift * torch.eye(s, device=device, dtype=dtype)
        eigvals2, eigvecs2 = torch.linalg.eigh(QtY_pd)
        eigvals2 = eigvals2.clamp_min(1e-20)
        R = eigvecs2 @ torch.diag(eigvals2.rsqrt()) @ eigvecs2.T
        B = Y_nu @ R

    # Thin SVD
    U_hat, Sigma, _ = torch.linalg.svd(B, full_matrices=False)
    Lambda_hat = torch.clamp(Sigma**2 - (nu + lam_shift), min=0.0)

    return NystromPreconditioner(U=U_hat, eigs=Lambda_hat, sketch_size=s)
