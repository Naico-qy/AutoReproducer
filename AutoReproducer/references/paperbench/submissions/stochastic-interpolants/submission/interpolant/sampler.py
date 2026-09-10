"""Inference-time samplers — Algorithm 2 of the paper.

Two integrators are exposed:

* :func:`sample_ode_euler` — exactly Algorithm 2 (forward Euler).
* :func:`sample_ode_dopri` — adaptive Dormand-Prince via torchdiffeq,
  matching Appendix B's "Dopri solver from the torchdiffeq library".

Both integrate the probability flow ODE  Ẋ_t = b̂_t(X_t, ξ)  from t=0 to
t=1, starting from X_0 = m(x_1) + σ ζ (which is sampled by the coupling).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
# (x: (B,C,H,W), t: (B,)) -> velocity (B,C,H,W)


# ---------------------------------------------------------------------------
# Forward Euler — Algorithm 2 ------------------------------------------------
# ---------------------------------------------------------------------------


def sample_ode_euler(
    velocity_fn: VelocityFn,
    x0: torch.Tensor,
    n_steps: int = 100,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> torch.Tensor:
    """Algorithm 2: X_{i+1} = X_i + N⁻¹ b̂_{i/N}(X_i)."""
    B = x0.shape[0]
    device = x0.device
    dtype = x0.dtype
    x = x0.clone()
    dt = (t_end - t_start) / n_steps
    ts = torch.linspace(t_start, t_end, n_steps + 1, device=device, dtype=dtype)
    for i in range(n_steps):
        t_b = ts[i].expand(B)
        v = velocity_fn(x, t_b)
        x = x + dt * v
    return x


# ---------------------------------------------------------------------------
# Dopri5 via torchdiffeq -----------------------------------------------------
# ---------------------------------------------------------------------------


def sample_ode_dopri(
    velocity_fn: VelocityFn,
    x0: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> torch.Tensor:
    """Adaptive integrator (Appendix B uses dopri from torchdiffeq)."""
    try:
        from torchdiffeq import odeint
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torchdiffeq is required for the Dopri sampler; "
            "install it with `pip install torchdiffeq`."
        ) from exc

    B = x0.shape[0]
    shape = x0.shape

    def rhs(t: torch.Tensor, x_flat: torch.Tensor) -> torch.Tensor:
        x_img = x_flat.view(shape)
        t_b = t.expand(B)
        v = velocity_fn(x_img, t_b)
        return v.reshape(x_flat.shape)

    t_span = torch.tensor([t_start, t_end], device=x0.device, dtype=x0.dtype)
    out = odeint(
        rhs,
        x0.reshape(B, -1),
        t_span,
        rtol=rtol,
        atol=atol,
        method="dopri5",
    )
    return out[-1].view(shape)


# ---------------------------------------------------------------------------
# Optional SDE samplers — Eq. (11) and (13) ----------------------------------
# ---------------------------------------------------------------------------


def sample_sde_euler_maruyama_forward(
    velocity_fn: VelocityFn,
    score_fn: Optional[VelocityFn],
    gamma_fn: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    eps_t: float = 0.0,
    n_steps: int = 100,
) -> torch.Tensor:
    """Forward SDE — Eq. (11):

        dX^F = b_t(X^F) dt - eps_t γ_t⁻¹ g_t(X^F) dt + √(2 eps_t) dW.

    With eps_t = 0 it reduces to the probability-flow ODE.
    """
    B = x0.shape[0]
    device = x0.device
    dtype = x0.dtype
    x = x0.clone()
    dt = 1.0 / n_steps
    sqrt_dt = dt**0.5
    for i in range(n_steps):
        t_b = torch.full((B,), i * dt, device=device, dtype=dtype)
        v = velocity_fn(x, t_b)
        drift = v
        if eps_t > 0.0 and score_fn is not None:
            g = score_fn(x, t_b)
            gamma_t = gamma_fn(t_b).view(-1, *([1] * (x.dim() - 1)))
            drift = drift - eps_t * (g / torch.clamp(gamma_t, min=1e-6))
            x = x + drift * dt + (2.0 * eps_t) ** 0.5 * sqrt_dt * torch.randn_like(x)
        else:
            x = x + drift * dt
    return x
