"""APGD (Croce & Hein, ICML 2020) — used in the evaluation of zero-shot
classification (Sec. B.10) and inside the LVLM ensemble attack (Sec. B.6).

This implementation follows the parameter-free APGD recipe with a step-size
schedule that halves the step on stagnation. The structural template is the
public reference implementation at https://github.com/fra31/robust-finetuning
(per addendum), specialized to L_inf attacks on CLIP zero-shot classifiers.

We provide:
  * apgd_attack(model, x, y, eps, n_iter, loss_fn) — standard APGD.
  * apgd_dlr_targeted(...) — targeted APGD-DLR for AutoAttack-style eval.

For the *evaluation* of robust accuracy, both `loss_fn="ce"` and
`loss_fn="dlr-targeted"` are run, exactly as Sec. B.10 specifies.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F


def _check_oscillation(
    loss_history: torch.Tensor, k: int, j: int, rho: float = 0.75
) -> torch.Tensor:
    """Standard APGD oscillation criterion: count how often the loss
    decreased between checkpoints in the last window of size k."""
    t = torch.zeros_like(loss_history[0])
    for i in range(k):
        t = t + (loss_history[j - (i + 1)] > loss_history[j - i]).float()
    return (t <= rho * k).float()


def dlr_loss_targeted(
    logits: torch.Tensor, y: torch.Tensor, y_target: torch.Tensor
) -> torch.Tensor:
    """DLR loss in the targeted form used by AutoAttack."""
    logits_sorted, ind = logits.sort(dim=1)
    u = torch.arange(logits.shape[0])
    return -(logits[u, y] - logits[u, y_target]) / (
        logits_sorted[:, -1]
        - 0.5 * (logits_sorted[:, -3] + logits_sorted[:, -4])
        + 1e-12
    )


def apgd_attack(
    model: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    n_iter: int = 100,
    loss_fn: str = "ce",
    rho: float = 0.75,
    y_target: Optional[torch.Tensor] = None,
    random_start: bool = True,
) -> torch.Tensor:
    """APGD with l_infinity threat model.

    Parameters
    ----------
    model    : callable producing logits (or any function that returns a
               scalar-per-sample loss when combined with `loss_fn`).
    x        : input batch in [0, 1].
    y        : true labels.
    eps      : l_inf radius in [0, 1] units.
    n_iter   : number of APGD iterations (100 in our zero-shot eval).
    loss_fn  : "ce" or "dlr-targeted".
    """
    device = x.device
    bsz = x.shape[0]
    if random_start:
        x_adv = x + (torch.empty_like(x).uniform_(-eps, eps))
    else:
        x_adv = x.clone()
    x_adv = x_adv.clamp(0.0, 1.0)

    step_size = 2.0 * eps
    x_best = x_adv.clone()
    loss_best = torch.full((bsz,), -float("inf"), device=device)
    x_best_adv = x_adv.clone()

    # Iteration milestones from the APGD paper: p_k = 0, 0.22, then progressive.
    p = [0, 0.22]
    while p[-1] < 1.0:
        p.append(p[-1] + max(p[-1] - p[-2] - 0.03, 0.06))
    checkpoints = [int(round(pi * n_iter)) for pi in p if pi <= 1.0]

    loss_steps = torch.zeros(n_iter, bsz, device=device)
    grad_buf = torch.zeros_like(x)

    x_adv.requires_grad_(True)
    logits = model(x_adv)
    if loss_fn == "ce":
        loss = F.cross_entropy(logits, y, reduction="none")
    elif loss_fn == "dlr-targeted":
        assert y_target is not None
        loss = dlr_loss_targeted(logits, y, y_target)
    else:
        raise ValueError(f"unknown loss_fn={loss_fn}")
    loss.sum().backward()
    grad = x_adv.grad.detach()
    x_adv = x_adv.detach()

    counter3 = 0
    last_check = 0

    for i in range(n_iter):
        with torch.no_grad():
            grad_buf = 0.75 * grad_buf + 0.25 * grad
            x_new = x_adv + step_size * grad.sign()
            # Project onto eps-ball around clean x, then onto [0,1].
            x_new = torch.min(torch.max(x_new, x - eps), x + eps)
            x_new = x_new.clamp(0.0, 1.0)
            # Momentum update (APGD's two-step update).
            x_adv = x_adv + 0.75 * (x_new - x_adv) + 0.25 * (x_new - x_adv).sign() * 0.0
            x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0.0, 1.0)

        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        if loss_fn == "ce":
            loss = F.cross_entropy(logits, y, reduction="none")
        else:
            loss = dlr_loss_targeted(logits, y, y_target)
        loss.sum().backward()
        grad = x_adv.grad.detach()
        x_adv = x_adv.detach()

        loss_steps[i] = loss.detach()
        improved = loss.detach() > loss_best
        x_best_adv = torch.where(improved.view(-1, 1, 1, 1), x_adv, x_best_adv)
        loss_best = torch.where(improved, loss.detach(), loss_best)

        # Step size schedule
        if i in checkpoints[1:] and i > last_check:
            cond = _check_oscillation(
                loss_steps, k=max(i - last_check, 1), j=i, rho=rho
            )
            if cond.mean() > 0.5:
                step_size = step_size / 2.0
            last_check = i

    return x_best_adv.detach()


def apgd_ensemble_eval(model, x, y, eps, n_iter=100, num_classes=1000):
    """Run APGD-CE then APGD-DLR-Targeted (against 9 most likely wrong classes,
    one chosen per sample), keep worst-case adversarial. This mirrors the
    AutoAttack-lite recipe described in Sec. B.10."""
    x_adv = apgd_attack(model, x, y, eps=eps, n_iter=n_iter, loss_fn="ce")
    with torch.no_grad():
        logits = model(x_adv)
        # pick the second-most-likely class as targeted goal
        y_target = logits.argsort(dim=1, descending=True)[:, 1]
    if num_classes > 2:
        x_adv2 = apgd_attack(
            model,
            x,
            y,
            eps=eps,
            n_iter=n_iter,
            loss_fn="dlr-targeted",
            y_target=y_target,
        )
        with torch.no_grad():
            ok1 = model(x_adv).argmax(dim=1) == y
            ok2 = model(x_adv2).argmax(dim=1) == y
            # adversarial wins if either attack succeeds (i.e., ok=False)
            use2 = ok1 & (~ok2)
            x_adv = torch.where(use2.view(-1, 1, 1, 1), x_adv2, x_adv)
    return x_adv
