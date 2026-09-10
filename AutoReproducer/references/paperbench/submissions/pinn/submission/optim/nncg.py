"""NysNewton-CG (NNCG) — Algorithm 4 of Rathore et al. (2024).

NNCG is a damped Newton-type optimizer that:
    (1) every F iterations, refreshes a Nyström low-rank preconditioner
        of the Hessian (Algorithm 5);
    (2) computes a damped-Newton search direction
            d_k = (H + μ I)^{-1} ∇L
        via Nyström-preconditioned CG (Algorithm 6) — warm-started from
        the previous step's direction;
    (3) selects a step size by Armijo backtracking line search
        (Algorithm 7).

Default hyperparameters from Appendix E.2 of the paper:

    η = 1, K = 2000, s = 60, F = 20, ε = 1e-16, M = 1000,
    α = 0.1, β = 0.5,  μ ∈ {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}
    (the authors find μ ∈ {1e-2, 1e-1} works best).

API mirrors torch.optim.Optimizer's `.step(closure)` signature so it can
be used as a drop-in replacement after Adam+L-BFGS.

The Hessian-vector product is computed via torch.autograd.grad at the
current parameters (Pearlmutter 1994).
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional

import torch
from torch.optim.optimizer import Optimizer

from .nystrom import (
    NystromPreconditioner,
    randomized_nystrom_approximation,
)
from .pcg import nystrom_pcg


def _flatten_params(params: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in params])


def _set_params(params: List[torch.Tensor], flat: torch.Tensor) -> None:
    offset = 0
    for p in params:
        n = p.numel()
        p.data.copy_(flat[offset : offset + n].view_as(p))
        offset += n


def _flatten_grads(params: List[torch.Tensor]) -> torch.Tensor:
    grads = []
    for p in params:
        if p.grad is None:
            grads.append(torch.zeros_like(p).reshape(-1))
        else:
            grads.append(p.grad.detach().reshape(-1))
    return torch.cat(grads)


class NysNewtonCG(Optimizer):
    """NysNewton-CG optimizer (Algorithm 4).

    Parameters
    ----------
    params : iterable of torch.nn.Parameter
        Model parameters.
    lr : float, default 1.0
        Initial step size η passed to Armijo backtracking.
    mu : float, default 1e-2
        Damping parameter μ for the (H + μ I) system.  In the paper,
        μ is tuned over {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}; the authors
        recommend μ ∈ {1e-2, 1e-1} (Appendix E.2).
    sketch_size : int, default 60
        Nyström sketch size s.
    update_freq : int, default 20
        Refresh the Nyström preconditioner every F iterations.
    cg_tol : float, default 1e-16
        CG tolerance ε.
    cg_max_iter : int, default 1000
        CG maximum iterations M.
    armijo_alpha : float, default 0.1
        Sufficient-decrease constant α.
    armijo_beta : float, default 0.5
        Backtracking factor β.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1.0,
        mu: float = 1e-2,
        sketch_size: int = 60,
        update_freq: int = 20,
        cg_tol: float = 1e-16,
        cg_max_iter: int = 1000,
        armijo_alpha: float = 0.1,
        armijo_beta: float = 0.5,
    ) -> None:
        defaults = dict(
            lr=lr,
            mu=mu,
            sketch_size=sketch_size,
            update_freq=update_freq,
            cg_tol=cg_tol,
            cg_max_iter=cg_max_iter,
            armijo_alpha=armijo_alpha,
            armijo_beta=armijo_beta,
        )
        super().__init__(params, defaults)
        self._iter_count = 0
        self._d_prev: Optional[torch.Tensor] = None
        self._prec: Optional[NystromPreconditioner] = None

    # ------------------------------------------------------------------
    # Hessian-vector product helper
    # ------------------------------------------------------------------
    @staticmethod
    def _hvp_factory(
        loss: torch.Tensor,
        params: List[torch.Tensor],
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return a closure v -> H v evaluated at the current parameters.

        Re-uses the autograd graph of `loss`: we differentiate once with
        create_graph=True to obtain the gradient, then use a second
        autograd call inside the returned closure.  This implements
        Pearlmutter (1994).
        """
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grad = torch.cat([g.reshape(-1) for g in grads])

        def hvp(v: torch.Tensor) -> torch.Tensor:
            gv = torch.autograd.grad(
                flat_grad, params, grad_outputs=v, retain_graph=True
            )
            return torch.cat([g.reshape(-1) for g in gv]).detach()

        return hvp

    # ------------------------------------------------------------------
    # Armijo (Algorithm 7)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _armijo(
        self,
        closure: Callable[[], torch.Tensor],
        params: List[torch.Tensor],
        flat0: torch.Tensor,
        grad: torch.Tensor,
        d: torch.Tensor,
        f0: float,
        eta: float,
        alpha: float,
        beta: float,
        max_ls_steps: int = 25,
    ) -> float:
        """Backtracking Armijo line search.

        Search direction is `-d` (we minimize), so the directional
        derivative is ∇L · (−d) = −grad·d.
        """
        gd = -float(torch.dot(grad, d))  # = ∇L · (−d)
        t = float(eta)
        for _ in range(max_ls_steps):
            _set_params(params, flat0 + t * (-d))
            with torch.enable_grad():
                f_new = float(closure().detach())
            if f_new <= f0 + alpha * t * gd:
                return t
            t *= beta
        # Fallback to a tiny step if line search fails.
        _set_params(params, flat0 + t * (-d))
        return t

    # ------------------------------------------------------------------
    # Main step (Algorithm 4)
    # ------------------------------------------------------------------
    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        if len(self.param_groups) != 1:
            raise ValueError("NysNewtonCG expects a single param group.")
        group = self.param_groups[0]
        params: List[torch.Tensor] = [p for p in group["params"] if p.requires_grad]

        # -- 1. Evaluate loss with create_graph=True to enable HVPs.
        with torch.enable_grad():
            loss = closure()
            hvp = self._hvp_factory(loss, params)
            grads = torch.autograd.grad(loss, params, retain_graph=True)
            flat_grad = torch.cat([g.detach().reshape(-1) for g in grads])

        device = flat_grad.device
        dtype = flat_grad.dtype
        p_dim = flat_grad.numel()
        f0 = float(loss.detach())

        # -- 2. Refresh the Nyström preconditioner every F iters.
        if self._iter_count % group["update_freq"] == 0 or self._prec is None:
            self._prec = randomized_nystrom_approximation(
                hvp,
                p=p_dim,
                s=group["sketch_size"],
                device=device,
                dtype=dtype,
            )

        # -- 3. Damped-Newton step via NyströmPCG, warm-started.
        if self._d_prev is None or self._d_prev.numel() != p_dim:
            x0 = torch.zeros_like(flat_grad)
        else:
            x0 = self._d_prev.to(device=device, dtype=dtype)

        d = nystrom_pcg(
            hvp=hvp,
            b=flat_grad.detach(),
            x0=x0,
            prec=self._prec,
            mu=group["mu"],
            tol=group["cg_tol"],
            max_iter=group["cg_max_iter"],
        )
        self._d_prev = d.detach().clone()

        # -- 4. Armijo line search to update parameters.
        flat0 = _flatten_params(params).clone()
        self._armijo(
            closure=closure,
            params=params,
            flat0=flat0,
            grad=flat_grad.detach(),
            d=d,
            f0=f0,
            eta=group["lr"],
            alpha=group["armijo_alpha"],
            beta=group["armijo_beta"],
        )

        self._iter_count += 1
        return loss.detach()
