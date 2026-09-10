"""PDE residual operators and loss for PINNs.

Implements the three PDEs studied in Rathore et al. (2024), Appendix A:

    Convection (A.1)
        ∂u/∂t + β ∂u/∂x = 0,  x∈(0, 2π), t∈(0, 1)
        u(x, 0) = sin(x);   u(0, t) = u(2π, t)
        β = 40.
        Analytical solution: u(x, t) = sin(x − β t).

    Reaction (A.2)
        ∂u/∂t − ρ u (1 − u) = 0,  x∈(0, 2π), t∈(0, 1)
        u(x, 0) = exp(−(x − π)² / (2 (π/4)²))
        u(0, t) = u(2π, t)
        ρ = 5.
        Analytical solution:
            h(x) = exp(−(x − π)² / (2 (π/4)²))
            u(x, t) = h(x) e^{ρ t} / (h(x) e^{ρ t} + 1 − h(x))

    Wave (A.3)
        ∂²u/∂t² − 4 ∂²u/∂x² = 0,  x∈(0, 1), t∈(0, 1)
        u(x, 0) = sin(π x) + 0.5 sin(β π x)
        ∂u/∂t (x, 0) = 0
        u(0, t) = u(1, t) = 0
        β = 5.
        Analytical solution:
            u(x, t) = sin(π x) cos(2 π t)
                     + 0.5 sin(β π x) cos(2 β π t)

The PINN training loss (Eq. 2 of the paper) is

    L(w) = (1 / 2 n_res) Σ (D[u(x_r;w)])²
         + (1 / 2 n_bc)  Σ (B[u(x_b;w)])²

where here we lump together initial-condition + boundary-condition points
into a single "boundary" set with the corresponding operators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Autograd helpers (Pearlmutter-style: one .grad() call per derivative)
# ---------------------------------------------------------------------------


def _grad(out: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """Compute d out / d inp summed over `out` axes (vectorized scalar field)."""
    g = torch.autograd.grad(
        out, inp, grad_outputs=torch.ones_like(out), create_graph=True
    )[0]
    return g


# ---------------------------------------------------------------------------
# PDE-specific residual computations
# ---------------------------------------------------------------------------


@dataclass
class PDESpec:
    """Bundle of (residual, ic, bc, exact_solution) callables for a PDE.

    All callables accept a torch model `u` and tensors of input
    coordinates (with requires_grad=True for residual / ic / bc), and
    return a tensor of pointwise values whose squared mean is the
    corresponding loss component.
    """

    name: str
    domain: Dict[str, float]  # x_low, x_high, t_low, t_high
    residual_fn: Callable[..., torch.Tensor]
    ic_fn: Callable[..., torch.Tensor]
    bc_fn: Callable[..., torch.Tensor]
    exact_fn: Callable[[torch.Tensor], torch.Tensor]
    # Whether the PDE has a second-order time derivative (wave)
    second_order_t: bool = False
    # Whether boundary condition is "periodic" (convection, reaction) or
    # "dirichlet zero" (wave)
    bc_type: str = "periodic"
    # Whether there is a derivative initial condition (wave: u_t(x,0)=0)
    has_velocity_ic: bool = False


# --- Convection (β = 40) -----------------------------------------------------


def convection_residual(
    u: nn.Module, xt: torch.Tensor, beta: float = 40.0
) -> torch.Tensor:
    xt = xt.requires_grad_(True)
    out = u(xt)
    grads = _grad(out, xt)  # (N, 2): d/dx, d/dt
    u_x, u_t = grads[:, 0:1], grads[:, 1:2]
    return u_t + beta * u_x


def convection_ic(u: nn.Module, x0: torch.Tensor) -> torch.Tensor:
    # x0 has shape (N, 2) with t==0
    return u(x0) - torch.sin(x0[:, 0:1])


def convection_bc(
    u: nn.Module, x_left: torch.Tensor, x_right: torch.Tensor
) -> torch.Tensor:
    # Periodic: u(0, t) − u(2π, t) = 0
    return u(x_left) - u(x_right)


def convection_exact(xt: torch.Tensor, beta: float = 40.0) -> torch.Tensor:
    return torch.sin(xt[:, 0:1] - beta * xt[:, 1:2])


# --- Reaction (ρ = 5) --------------------------------------------------------


def reaction_residual(u: nn.Module, xt: torch.Tensor, rho: float = 5.0) -> torch.Tensor:
    xt = xt.requires_grad_(True)
    out = u(xt)
    grads = _grad(out, xt)
    u_t = grads[:, 1:2]
    return u_t - rho * out * (1.0 - out)


def _reaction_h(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-((x - math.pi) ** 2) / (2.0 * (math.pi / 4.0) ** 2))


def reaction_ic(u: nn.Module, x0: torch.Tensor) -> torch.Tensor:
    return u(x0) - _reaction_h(x0[:, 0:1])


def reaction_bc(
    u: nn.Module, x_left: torch.Tensor, x_right: torch.Tensor
) -> torch.Tensor:
    return u(x_left) - u(x_right)


def reaction_exact(xt: torch.Tensor, rho: float = 5.0) -> torch.Tensor:
    h = _reaction_h(xt[:, 0:1])
    e = torch.exp(rho * xt[:, 1:2])
    return h * e / (h * e + 1.0 - h)


# --- Wave (β = 5, c = 2) -----------------------------------------------------


def wave_residual(u: nn.Module, xt: torch.Tensor, c2: float = 4.0) -> torch.Tensor:
    xt = xt.requires_grad_(True)
    out = u(xt)
    g = _grad(out, xt)
    u_x, u_t = g[:, 0:1], g[:, 1:2]
    u_xx = _grad(u_x, xt)[:, 0:1]
    u_tt = _grad(u_t, xt)[:, 1:2]
    return u_tt - c2 * u_xx


def wave_ic_displacement(
    u: nn.Module, x0: torch.Tensor, beta: float = 5.0
) -> torch.Tensor:
    target = torch.sin(math.pi * x0[:, 0:1]) + 0.5 * torch.sin(
        beta * math.pi * x0[:, 0:1]
    )
    return u(x0) - target


def wave_ic_velocity(u: nn.Module, x0: torch.Tensor) -> torch.Tensor:
    x0 = x0.requires_grad_(True)
    out = u(x0)
    u_t = _grad(out, x0)[:, 1:2]
    return u_t  # u_t(x, 0) == 0


def wave_bc(u: nn.Module, x_left: torch.Tensor, x_right: torch.Tensor) -> torch.Tensor:
    # Dirichlet zero on both sides: u(0, t) = u(1, t) = 0
    return torch.cat([u(x_left), u(x_right)], dim=0)


def wave_exact(xt: torch.Tensor, beta: float = 5.0) -> torch.Tensor:
    x = xt[:, 0:1]
    t = xt[:, 1:2]
    return torch.sin(math.pi * x) * torch.cos(2 * math.pi * t) + 0.5 * torch.sin(
        beta * math.pi * x
    ) * torch.cos(2 * beta * math.pi * t)


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------


def build_pde(name: str, **kwargs) -> PDESpec:
    name = name.lower()
    if name == "convection":
        beta = kwargs.get("beta", 40.0)
        return PDESpec(
            name="convection",
            domain={"x_low": 0.0, "x_high": 2 * math.pi, "t_low": 0.0, "t_high": 1.0},
            residual_fn=lambda u, xt: convection_residual(u, xt, beta=beta),
            ic_fn=convection_ic,
            bc_fn=convection_bc,
            exact_fn=lambda xt: convection_exact(xt, beta=beta),
            bc_type="periodic",
        )
    if name == "reaction":
        rho = kwargs.get("rho", 5.0)
        return PDESpec(
            name="reaction",
            domain={"x_low": 0.0, "x_high": 2 * math.pi, "t_low": 0.0, "t_high": 1.0},
            residual_fn=lambda u, xt: reaction_residual(u, xt, rho=rho),
            ic_fn=reaction_ic,
            bc_fn=reaction_bc,
            exact_fn=lambda xt: reaction_exact(xt, rho=rho),
            bc_type="periodic",
        )
    if name == "wave":
        beta = kwargs.get("beta", 5.0)
        c2 = kwargs.get("c2", 4.0)
        return PDESpec(
            name="wave",
            domain={"x_low": 0.0, "x_high": 1.0, "t_low": 0.0, "t_high": 1.0},
            residual_fn=lambda u, xt: wave_residual(u, xt, c2=c2),
            ic_fn=lambda u, x0: wave_ic_displacement(u, x0, beta=beta),
            bc_fn=wave_bc,
            exact_fn=lambda xt: wave_exact(xt, beta=beta),
            second_order_t=True,
            bc_type="dirichlet_zero",
            has_velocity_ic=True,
        )
    raise ValueError(f"Unknown PDE: {name}")


# ---------------------------------------------------------------------------
# Total PINN loss (Eq. 2)
# ---------------------------------------------------------------------------


def pinn_loss(
    model: nn.Module,
    pde: PDESpec,
    x_res: torch.Tensor,
    x_ic: torch.Tensor,
    x_bc_left: torch.Tensor,
    x_bc_right: torch.Tensor,
    x_ic_velocity: Optional[torch.Tensor] = None,
    return_components: bool = False,
):
    """Computes L(w) = 1/(2 n_res) Σ res² + 1/(2 n_bc) Σ bc².

    Note: the paper sets λ=1 and lumps initial-condition + boundary-
    condition points into a single denominator n_bc (Eq. 2).  We follow
    the same convention: the IC, BC and (for wave) velocity-IC residuals
    are concatenated and divided by 2*n_bc.
    """
    res = pde.residual_fn(model, x_res)
    res_loss = 0.5 * torch.mean(res**2)

    bc_terms = []
    bc_terms.append(pde.ic_fn(model, x_ic))
    bc_terms.append(pde.bc_fn(model, x_bc_left, x_bc_right))
    if pde.has_velocity_ic and x_ic_velocity is not None:
        bc_terms.append(wave_ic_velocity(model, x_ic_velocity))
    bc_concat = torch.cat(bc_terms, dim=0)
    bc_loss = 0.5 * torch.mean(bc_concat**2)

    total = res_loss + bc_loss
    if return_components:
        return total, {"residual": res_loss.detach(), "bc": bc_loss.detach()}
    return total
