"""PGD attacks used inside FARE / TeCoA training.

Per the paper (Sec. B.1) and addendum:
  * 10 PGD steps, alpha = 1/255, eps in {2/255, 4/255}
  * gradient normalization with elementwise sign for l_infinity
  * momentum factor of 0.9
  * uniform random initialization within the eps ball
  * the l_infinity ball is computed around the *non-normalized* input,
    i.e. in pixel space [0, 1]

The FARE inner-maximization solves
    z* = argmax_{||z - x||_inf <= eps}  || phi_FT(z) - phi_Org(x) ||_2^2
which we approximate with PGD on the trainable encoder phi_FT, holding
phi_Org(x) fixed (since it is frozen, its gradient is zero anyway).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _project_linf(delta: torch.Tensor, eps: float) -> torch.Tensor:
    """Project delta onto the l_infinity ball of radius eps."""
    return delta.clamp_(min=-eps, max=eps)


def _clip_to_image(x: torch.Tensor) -> torch.Tensor:
    """Clamp pixel values to [0, 1]."""
    return x.clamp_(min=0.0, max=1.0)


def pgd_fare(
    finetune,
    original,
    x: torch.Tensor,
    eps: float,
    alpha: float,
    num_steps: int = 10,
    momentum: float = 0.9,
    random_start: bool = True,
) -> torch.Tensor:
    """PGD inner maximizer for the FARE objective.

    Maximizes  || phi_FT(x + delta) - phi_Org(x) ||_2^2  over  ||delta||_inf <= eps.

    Returns the perturbed input z = x + delta in pixel space [0, 1].
    """
    finetune.eval()  # disable BN/dropout state changes during the inner loop
    with torch.no_grad():
        phi_org = original(x)

    if random_start:
        delta = torch.empty_like(x).uniform_(-eps, eps)
    else:
        delta = torch.zeros_like(x)

    delta = (x + delta).clamp_(0.0, 1.0) - x  # ensure x+delta in [0,1]
    delta.requires_grad_(True)

    grad_buf = torch.zeros_like(x)

    for _ in range(num_steps):
        phi_ft = finetune(x + delta)
        # Squared L2 distance per sample, then sum so .backward() yields per-sample grads.
        loss = (phi_ft - phi_org).pow(2).sum(dim=-1).sum()
        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[
            0
        ]
        # Sign-of-gradient with momentum (per addendum).
        grad_buf = momentum * grad_buf + grad
        step = alpha * grad_buf.sign()
        delta = delta.detach() + step
        # Project into the l_infinity ball, then clamp into [0, 1].
        delta = _project_linf(delta, eps)
        delta = (x + delta).clamp_(0.0, 1.0) - x
        delta.requires_grad_(True)

    finetune.train()
    return (x + delta).detach()


def pgd_tecoa(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    num_steps: int = 10,
    momentum: float = 0.9,
    random_start: bool = True,
) -> torch.Tensor:
    """PGD inner maximizer for the supervised TeCoA objective
    (Mao et al., 2023). Maximizes the cross-entropy loss against the
    true class y of zero-shot CLIP. Implemented for completeness so that
    the same training infrastructure can produce TeCoA baselines (see App.
    B.5 of the paper, where the authors compare TeCoA to FARE).
    """
    model.eval()
    if random_start:
        delta = torch.empty_like(x).uniform_(-eps, eps)
    else:
        delta = torch.zeros_like(x)
    delta = (x + delta).clamp_(0.0, 1.0) - x
    delta.requires_grad_(True)
    grad_buf = torch.zeros_like(x)

    for _ in range(num_steps):
        logits = model(x + delta)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, delta)[0]
        grad_buf = momentum * grad_buf + grad
        delta = delta.detach() + alpha * grad_buf.sign()
        delta = _project_linf(delta, eps)
        delta = (x + delta).clamp_(0.0, 1.0) - x
        delta.requires_grad_(True)

    model.train()
    return (x + delta).detach()
