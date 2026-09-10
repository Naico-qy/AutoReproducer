"""NyströmPCG (Algorithm 6, Appendix E.2 of Rathore et al. 2024).

Solves the damped Newton system

    (A + μ I) x = b

via preconditioned conjugate gradient with Nyström preconditioner

    P^{-1} = (λ̂_s + μ) U (Λ̂ + μ I)^{-1} U^T + (I − U U^T)

where (U, Λ̂) is the Nyström approximation of A.  λ̂_s denotes the
s-th (smallest in the kept top-s) approximate eigenvalue.

Hessian-vector products are exposed via callable hvp(v) -> torch.Tensor.
"""

from __future__ import annotations

from typing import Callable

import torch

from .nystrom import NystromPreconditioner


def _apply_Pinv(
    r: torch.Tensor, prec: NystromPreconditioner, mu: float
) -> torch.Tensor:
    """Apply P^{-1} to vector r.

    P^{-1} r = (λ̂_s + μ) U (Λ̂ + μ I)^{-1} U^T r + (I − U U^T) r
    """
    U, eigs = prec.U, prec.eigs
    lam_s = (
        eigs[-1]
        if eigs.numel() > 0
        else torch.zeros((), dtype=r.dtype, device=r.device)
    )
    Utr = U.T @ r  # (s,)
    scaled = ((lam_s + mu) / (eigs + mu)) * Utr  # (s,)
    return U @ scaled + (r - U @ Utr)


def nystrom_pcg(
    hvp: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: torch.Tensor,
    prec: NystromPreconditioner,
    mu: float,
    tol: float = 1e-16,
    max_iter: int = 1000,
) -> torch.Tensor:
    """Nyström-preconditioned conjugate gradient (Algorithm 6).

    Solves (A + μ I) x = b approximately.

    Notes
    -----
    The CG state vectors are kept on the same device / dtype as `b`.
    `hvp(v)` should return A v (not (A + μ I) v).
    """
    x = x0.clone()

    def Av_damped(v: torch.Tensor) -> torch.Tensor:
        return hvp(v) + mu * v

    r = b - Av_damped(x)
    z = _apply_Pinv(r, prec, mu)
    p = z.clone()
    rz = torch.dot(r, z)
    b_norm = torch.linalg.norm(b).clamp_min(1e-30)

    for k in range(max_iter):
        if torch.linalg.norm(r) < tol * b_norm:
            break
        Ap = Av_damped(p)
        pAp = torch.dot(p, Ap).clamp_min(1e-30)
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        z = _apply_Pinv(r, prec, mu)
        rz_new = torch.dot(r, z)
        beta = rz_new / rz.clamp_min(1e-30)
        p = z + beta * p
        rz = rz_new
    return x
