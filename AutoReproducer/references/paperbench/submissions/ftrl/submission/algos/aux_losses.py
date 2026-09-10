"""Knowledge-retention auxiliary losses (paper §2 / Appendix C).

The paper considers three retention auxiliary losses, all added on top of the
underlying RL loss with a configurable coefficient `lambda`:

* **BC** - Behavioral Cloning replay (App. C.2):

      L_BC(θ) = E_{s ∼ B_BC} [ KL( π_θ(·|s) ‖ π*(·|s) ) ]

  where `B_BC` is a buffer of expert states (collected before fine-tuning).
  Following Wolczyk et al. (2022) we apply `KL(π_θ ‖ π*)` to keep the loss
  bounded and align with the cited code-base.

* **KS** - Kickstarting (App. C.2 / Schmitt et al. 2018):

      L_KS(θ) = E_{s ∼ π_θ} [ KL( π*(·|s) ‖ π_θ(·|s) ) ]

  The state distribution is the *online* one of the fine-tuned policy.

* **EWC** - Elastic Weight Consolidation (App. C.1 / Kirkpatrick et al. 2017):

      L_EWC(θ) = Σ_i F_i ( θ*_i - θ_i )^2

  with `F_i` the diagonal of the Fisher information matrix at `θ*` (estimated
  in `fisher.py`).
"""

from __future__ import annotations

from typing import Iterable, Mapping

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Distillation-based losses (BC, KS).
# ---------------------------------------------------------------------------


def kl_categorical(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Element-wise KL( softmax(p_logits) || softmax(q_logits) )."""
    p_logp = F.log_softmax(p_logits, dim=-1)
    q_logp = F.log_softmax(q_logits, dim=-1)
    p = p_logp.exp()
    return (p * (p_logp - q_logp)).sum(dim=-1)


def bc_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """L_BC = E_{s ∼ B_BC} KL( π_θ(·|s) || π*(·|s) ).

    Args
    ----
    student_logits : (..., A) action logits from the fine-tuned policy `π_θ`.
    teacher_logits : (..., A) action logits from the pre-trained policy `π*`.
    """
    return kl_categorical(student_logits, teacher_logits).mean()


def ks_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """L_KS = E_{s ∼ π_θ} KL( π*(·|s) || π_θ(·|s) ).

    The expectation is over states from online rollouts (the caller batches
    them); only the KL direction differs from `bc_loss`.
    """
    return kl_categorical(teacher_logits, student_logits).mean()


# ---------------------------------------------------------------------------
# Continuous-action distillation losses (used for SAC actor on Meta-World).
# ---------------------------------------------------------------------------


def gaussian_kl(
    p_mean: torch.Tensor,
    p_logstd: torch.Tensor,
    q_mean: torch.Tensor,
    q_logstd: torch.Tensor,
) -> torch.Tensor:
    """KL( N(p_mean, exp(p_logstd)^2) || N(q_mean, exp(q_logstd)^2) ) summed
    over action dim, averaged over batch."""
    p_var = (2 * p_logstd).exp()
    q_var = (2 * q_logstd).exp()
    kl = q_logstd - p_logstd + (p_var + (p_mean - q_mean).pow(2)) / (2 * q_var) - 0.5
    return kl.sum(dim=-1).mean()


def bc_loss_gaussian(
    student_mean: torch.Tensor,
    student_logstd: torch.Tensor,
    teacher_mean: torch.Tensor,
    teacher_logstd: torch.Tensor,
) -> torch.Tensor:
    return gaussian_kl(student_mean, student_logstd, teacher_mean, teacher_logstd)


# ---------------------------------------------------------------------------
# EWC quadratic anchor.
# ---------------------------------------------------------------------------


def ewc_loss(
    current_params: Iterable[torch.Tensor],
    pretrained_params: Mapping[str, torch.Tensor],
    fisher: Mapping[str, torch.Tensor],
    named_iterator: Iterable,
) -> torch.Tensor:
    """L_EWC = Σ_i F_i ( θ*_i - θ_i )^2.

    `named_iterator` should yield `(name, parameter)` pairs of the *current*
    model. `pretrained_params` and `fisher` are dicts keyed by the same names.
    """
    loss = torch.zeros((), device=next(iter(current_params)).device)
    for name, p in named_iterator:
        if name not in fisher:
            continue
        loss = loss + (fisher[name] * (p - pretrained_params[name]).pow(2)).sum()
    return loss


def ks_decay_schedule(initial: float, decay: float, step: int) -> float:
    """Exponential decay of the KS coefficient (App. B.1: 0.99998^step)."""
    return initial * (decay**step)
