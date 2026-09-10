"""Sampling routines for the Simformer.

Implements:
- ``sample_conditional``: solve the reverse SDE on latent coordinates while
  holding conditioned coordinates fixed at the observation (paper §3.3,
  following Weilbach et al. 2023).
- ``guided_sample``: diffusion guidance for inequality constraints
  c(x̂) ≤ 0 via the universal-guidance formula (Eq. 9 of the paper):

      s_φ(x̂_t, t | c) ≈ s_φ(x̂_t, t)
                       + ∇_{x̂_t} log σ(-s(t) c(x̂_t))
"""

from __future__ import annotations

import torch
from torch import Tensor

from .sde import BaseSDE, VESDE, VPSDE


@torch.no_grad()
def sample_conditional(
    model,
    sde: BaseSDE,
    condition_mask: Tensor,
    observed_values: Tensor,
    num_steps: int = 50,
    attention_mask: Tensor | None = None,
    num_samples: int | None = None,
) -> Tensor:
    """Solve the reverse SDE for latent coordinates.

    Parameters
    ----------
    condition_mask     : (N,) or (S, N) bool, True = observed.
    observed_values    : (N,) or (S, N) values for conditioned coords (latent
                         positions are ignored / overwritten).
    num_samples        : if condition_mask is 1-D, replicate to this many.
    """
    model.eval()
    device = next(model.parameters()).device

    if condition_mask.dim() == 1:
        if num_samples is None:
            raise ValueError("num_samples required for 1-D condition_mask.")
        condition_mask = (
            condition_mask.unsqueeze(0).expand(num_samples, -1).contiguous()
        )
        observed_values = (
            observed_values.unsqueeze(0).expand(num_samples, -1).contiguous()
        )

    S, N = condition_mask.shape
    cond = condition_mask.to(device).bool()
    obs = observed_values.to(device).float()

    # Initial noise
    x = sde.prior_sample((S, N), device=device)
    x = torch.where(cond, obs, x)

    # Linear time grid from T -> eps
    ts = torch.linspace(sde.T, sde.eps, num_steps + 1, device=device)
    dt = (sde.T - sde.eps) / num_steps

    for i in range(num_steps):
        t = ts[i].expand(S)
        score = model(x, cond, t, attention_mask)  # (S, N)
        drift = sde.drift(x, t)
        g = sde.diffusion(t)
        # Reverse-time SDE drift
        rev_drift = drift - (g**2) * score
        # Euler-Maruyama step (going backward)
        z = torch.randn_like(x)
        x_next = (
            x - rev_drift * dt + g * torch.sqrt(torch.tensor(dt, device=device)) * z
        )
        # Keep conditioned coords clamped at the observation.
        x = torch.where(cond, obs, x_next)
    return x


def guided_sample(
    model,
    sde: BaseSDE,
    condition_mask: Tensor,
    observed_values: Tensor,
    constraint_fn,
    num_steps: int = 50,
    guidance_scale_fn=None,
    attention_mask: Tensor | None = None,
    num_samples: int | None = None,
) -> Tensor:
    """Sample with universal diffusion guidance for c(x̂) ≤ 0.

    The guided score is

        s_guided = s_φ + ∇_x log σ(-s(t) * c(x̂))      (Eq. 9 in the paper).

    Parameters
    ----------
    constraint_fn   : callable, x -> (S,) tensor; satisfied iff <= 0.
    guidance_scale_fn : callable t -> scalar s(t); defaults to 1/std(t)^2 for
        VESDE so that s(t) -> ∞ as t -> 0 (paper §3.4).
    """
    model.eval()
    device = next(model.parameters()).device

    if condition_mask.dim() == 1:
        if num_samples is None:
            raise ValueError("num_samples required for 1-D condition_mask.")
        condition_mask = (
            condition_mask.unsqueeze(0).expand(num_samples, -1).contiguous()
        )
        observed_values = (
            observed_values.unsqueeze(0).expand(num_samples, -1).contiguous()
        )

    S, N = condition_mask.shape
    cond = condition_mask.to(device).bool()
    obs = observed_values.to(device).float()

    if guidance_scale_fn is None:
        # 1 / sigma(t)^2  for VESDE, 1/std(t)^2 for VPSDE; both ↑∞ as t↓0.
        def guidance_scale_fn(tt: Tensor) -> Tensor:
            _, std = sde.marginal(torch.zeros(1, device=device), tt)
            return 1.0 / (std**2 + 1e-6)

    x = sde.prior_sample((S, N), device=device)
    x = torch.where(cond, obs, x)

    ts = torch.linspace(sde.T, sde.eps, num_steps + 1, device=device)
    dt = (sde.T - sde.eps) / num_steps

    for i in range(num_steps):
        t = ts[i].expand(S)
        with torch.enable_grad():
            x_req = x.detach().clone().requires_grad_(True)
            c_val = constraint_fn(x_req)  # (S,)
            scale = guidance_scale_fn(t)
            log_sig = torch.nn.functional.logsigmoid(-scale * c_val)  # (S,)
            grad_guidance = torch.autograd.grad(log_sig.sum(), x_req)[0]
        with torch.no_grad():
            score = model(x, cond, t, attention_mask)
            score = score + grad_guidance
            drift = sde.drift(x, t)
            g = sde.diffusion(t)
            rev_drift = drift - (g**2) * score
            z = torch.randn_like(x)
            x_next = (
                x - rev_drift * dt + g * torch.sqrt(torch.tensor(dt, device=device)) * z
            )
            x = torch.where(cond, obs, x_next)
    return x.detach()
