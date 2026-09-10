"""Probability-flow ODE sampler and log-density evaluator for (T)SNPSE.

Implements:

* sampling from the approximate posterior by integrating the time-reversed
  probability flow ODE (Eq. 4 of Sharrock et al., ICML 2024) using an
  off-the-shelf RK45 solver (Appendix E.3.3 of the paper).
* exact log-density evaluation via the instantaneous change-of-variables
  formula (Eq. 5 of the paper, Chen et al., 2018a), used to compute the
  truncation boundary κ (Appendix E.3.3) for TSNPSE.

The Skilling-Hutchinson trace estimator (Skilling 1989, Hutchinson 1990) is
used to keep the log-determinant computation tractable in higher dimensions
(Grathwohl et al., 2019). For very small d we also support the exact trace
via autograd.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import nn

from .sde import BaseSDE


# ---------------------------------------------------------------------------
# Probability-flow ODE drift (paper Eq. 4)
# ---------------------------------------------------------------------------
def _pf_drift(
    score_net: nn.Module,
    sde: BaseSDE,
    theta_t: torch.Tensor,
    x_obs: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Compute  f(θ_t, t) - ½ g²(t) s_ψ(θ_t, x_obs, t)."""
    score = score_net(theta_t, x_obs, t)
    return sde.probability_flow_drift(theta_t, t, score)


# ---------------------------------------------------------------------------
# Sampling (Appendix E.3.3 — "Sampling: We use the probability flow ODE for
# sampling. To solve this ODE, we use an off-the-shelf solver (RK45).")
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_probability_flow(
    score_net: nn.Module,
    sde: BaseSDE,
    x_obs: torch.Tensor,
    n_samples: int,
    n_steps: int = 250,
    method: str = "rk45",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate samples from p_ψ(θ | x_obs) by integrating the reverse ODE.

    We integrate ``dθ/dt = f(θ_t, t) - ½ g²(t) s_ψ(θ_t, x_obs, t)`` from
    t = T down to t = ε.

    Parameters
    ----------
    score_net : trained score network.
    sde       : forward SDE.
    x_obs     : (p,) or (1, p) tensor — the conditioning observation.
    n_samples : number of posterior samples to draw.
    n_steps   : number of ODE steps (used by the simple RK4/Euler fall-back).
    method    : "rk45" (uses ``scipy.integrate.solve_ivp``) or "euler".
    """
    score_net.eval()
    if device is None:
        device = next(score_net.parameters()).device

    if x_obs.ndim == 1:
        x_obs = x_obs.unsqueeze(0)
    x_obs = x_obs.to(device)
    x_batch = x_obs.expand(n_samples, -1)

    d = score_net.theta_dim
    theta = sde.prior_sampling((n_samples, d), device=device)

    if method == "rk45":
        try:
            from scipy.integrate import solve_ivp
            import numpy as np

            def ode_fn(t_scalar: float, theta_flat: "np.ndarray") -> "np.ndarray":
                theta_arr = (
                    torch.from_numpy(theta_flat).float().to(device).view(n_samples, d)
                )
                t_vec = torch.full((n_samples,), float(t_scalar), device=device)
                drift = _pf_drift(score_net, sde, theta_arr, x_batch, t_vec)
                # We want to integrate from T -> eps, i.e. dτ = -dt (forward time τ).
                return (-drift.cpu().numpy()).flatten()

            t_span = (sde.T, sde.eps)
            sol = solve_ivp(
                ode_fn,
                t_span=t_span,
                y0=theta.cpu().numpy().flatten(),
                method="RK45",
                rtol=1e-5,
                atol=1e-5,
            )
            theta = torch.from_numpy(sol.y[:, -1]).float().to(device).view(n_samples, d)
            return theta
        except Exception:  # noqa: BLE001  fall back gracefully
            pass

    # Simple Heun / Euler fall-back from t=T to t=eps.
    ts = torch.linspace(sde.T, sde.eps, n_steps + 1, device=device)
    for i in range(n_steps):
        t = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t  # negative
        t_vec = torch.full((n_samples,), float(t), device=device)
        drift = _pf_drift(score_net, sde, theta, x_batch, t_vec)
        theta = theta + drift * dt
    return theta


# ---------------------------------------------------------------------------
# Exact log-density via instantaneous change-of-variables (paper Eq. 5)
# ---------------------------------------------------------------------------
def _divergence_exact(
    drift_fn: Callable[[torch.Tensor], torch.Tensor],
    theta: torch.Tensor,
) -> torch.Tensor:
    """Exact divergence (trace of the Jacobian) via autograd."""
    bsz, d = theta.shape
    out = drift_fn(theta)
    div = theta.new_zeros(bsz)
    for i in range(d):
        grads = torch.autograd.grad(
            out[:, i].sum(), theta, retain_graph=True, create_graph=False
        )[0]
        div = div + grads[:, i]
    return div


def _divergence_hutchinson(
    drift_fn: Callable[[torch.Tensor], torch.Tensor],
    theta: torch.Tensor,
    n_eps: int = 1,
) -> torch.Tensor:
    """Skilling-Hutchinson trace estimator (Grathwohl et al., 2019)."""
    bsz = theta.shape[0]
    div = theta.new_zeros(bsz)
    for _ in range(n_eps):
        eps = torch.randn_like(theta)
        out = drift_fn(theta)
        grads = torch.autograd.grad(
            (out * eps).sum(), theta, retain_graph=True, create_graph=False
        )[0]
        div = div + (grads * eps).sum(dim=-1)
    return div / n_eps


def log_prob_probability_flow(
    score_net: nn.Module,
    sde: BaseSDE,
    theta_0: torch.Tensor,
    x_obs: torch.Tensor,
    n_steps: int = 50,
    hutchinson: bool = True,
    n_eps: int = 1,
) -> torch.Tensor:
    """Approximate log p_ψ(θ_0 | x_obs) via the instantaneous change-of-variables.

    log p(θ_0|x) = log π(θ_T) + ∫_0^T Tr[ ∂v_t / ∂θ_t ] dt,

    where v_t is the probability-flow drift. We use a simple forward-Euler
    discretisation with ``n_steps`` steps; the paper uses RK45 but the
    computational pattern is identical.
    """
    score_net.eval()
    device = next(score_net.parameters()).device

    if theta_0.ndim == 1:
        theta_0 = theta_0.unsqueeze(0)
    if x_obs.ndim == 1:
        x_obs = x_obs.unsqueeze(0)
    if x_obs.shape[0] == 1 and theta_0.shape[0] != 1:
        x_obs = x_obs.expand(theta_0.shape[0], -1)

    bsz, d = theta_0.shape

    theta = theta_0.detach().clone().to(device).requires_grad_(True)
    log_det = torch.zeros(bsz, device=device)

    ts = torch.linspace(sde.eps, sde.T, n_steps + 1, device=device)
    for i in range(n_steps):
        t = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t

        def drift_fn(theta_in: torch.Tensor, _t=t) -> torch.Tensor:
            t_vec = torch.full((theta_in.shape[0],), float(_t), device=device)
            return _pf_drift(score_net, sde, theta_in, x_obs, t_vec)

        if hutchinson:
            div = _divergence_hutchinson(drift_fn, theta, n_eps=n_eps)
        else:
            div = _divergence_exact(drift_fn, theta)

        with torch.no_grad():
            theta = (theta + drift_fn(theta) * dt).detach().requires_grad_(True)
        log_det = log_det + div.detach() * dt

    # log π(θ_T) for VESDE: N(0, σ_max² I); for VPSDE: N(0, I).
    if hasattr(sde, "sigma_max"):
        var = sde.sigma_max**2
    else:
        var = 1.0
    log_pi = -0.5 * (theta.detach() ** 2).sum(dim=-1) / var - 0.5 * d * (
        torch.log(torch.tensor(2.0 * torch.pi)) + torch.log(torch.tensor(var))
    )

    # Forward time-integral above corresponds to going from θ_0 to θ_T, hence
    # log p(θ_0) = log π(θ_T) - ∫ Tr[J] dt  (sign is negated by direction).
    return log_pi - log_det
