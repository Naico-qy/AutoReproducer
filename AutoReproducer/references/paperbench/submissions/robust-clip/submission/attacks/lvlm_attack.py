"""Untargeted ensemble attack pipeline for LVLM evaluation (Sec. 4.1, App. B.6).

Pipeline (per paper):
  1. Half-precision (16-bit) APGD with 100 iterations against each of the
     ground-truth captions / answers.
  2. After each attack, drop samples already below score threshold
     (CIDEr<10 for COCO, <2 for Flickr30k, score==0 for VQA).
  3. Final: single-precision (32-bit) APGD using the ground truth that
     produced the lowest CIDEr/score so far, initialised from the running
     best perturbation. (Addendum confirms 16-bit / 32-bit precisions.)
  4. For VQA (and only-`maybe`-target on TextVQA), additional targeted attacks
     at single precision with target strings "maybe" and "Word". Targets
     start from a clean perturbation init.

We keep the *worst-case* score per sample across all attack stages.

We also implement the targeted attack used in Sec. 4.2 (10,000-iter APGD
against a specified target caption).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import torch

from .apgd import apgd_attack


@dataclass
class AttackResult:
    x_adv: torch.Tensor  # (B, 3, H, W) best adv sample so far
    best_score: torch.Tensor  # (B,) worst (=lowest) score seen
    best_gt_idx: torch.Tensor  # (B,) index of GT that produced worst score


def _score_fn_loss_wrapper(
    lvlm_loss_fn: Callable[[torch.Tensor, str], torch.Tensor],
    gt: str,
):
    """Wraps an LVLM teacher-forcing loss into a model-like callable so it
    can be used inside APGD.

    `lvlm_loss_fn(x, gt)` should return a per-sample scalar tensor of shape
    (B,) representing the negative log-likelihood of the ground-truth string
    `gt` given image x. Maximising this loss is the untargeted attack.
    """

    class _LossModel(torch.nn.Module):
        def forward(self, x):
            # Treat the loss as a single-class "logit": APGD will take CE
            # against y=0, which equals the loss itself.
            l = lvlm_loss_fn(x, gt)
            return torch.stack([l, -l], dim=1)  # (B, 2)

    return _LossModel()


def ensemble_untargeted_attack(
    lvlm_loss_fn: Callable,
    score_fn: Callable[[torch.Tensor, Sequence[str]], torch.Tensor],
    x: torch.Tensor,
    gts: List[List[str]],  # per sample list of ground-truth strings
    eps: float,
    score_threshold: float,
    n_iter_low: int = 100,
    n_iter_high: int = 100,
    half_precision: bool = True,
) -> AttackResult:
    """Run the half-then-single precision APGD ensemble attack."""
    B = x.shape[0]
    device = x.device
    best_x = x.clone()
    best_score = torch.full((B,), float("inf"), device=device)
    best_gt = torch.zeros((B,), dtype=torch.long, device=device)

    # Phase 1: half-precision APGD against each GT caption / answer.
    for gt_idx in range(max(len(g) for g in gts)):
        active = best_score > score_threshold
        if not active.any():
            break
        idx = active.nonzero(as_tuple=True)[0]
        x_act = x[idx]

        # One GT per active sample
        gt_for_active = [gts[int(i)][min(gt_idx, len(gts[int(i)]) - 1)] for i in idx]

        # Run with autocast(fp16) when half_precision=True (per addendum).
        ctx = torch.cuda.amp.autocast(enabled=half_precision)
        with ctx:
            # Build a stub "model" that returns the LVLM loss for *each* sample
            # against its own GT. We approximate by attacking sample-by-sample.
            adv_act_list = []
            for j, ig in zip(idx.tolist(), gt_for_active):
                stub = _score_fn_loss_wrapper(lvlm_loss_fn, ig)
                adv = apgd_attack(
                    stub,
                    x[j : j + 1],
                    torch.zeros(1, dtype=torch.long, device=device),
                    eps=eps,
                    n_iter=n_iter_low,
                    loss_fn="ce",
                )
                adv_act_list.append(adv)
        adv_act = torch.cat(adv_act_list, dim=0)

        # Score adv samples against ALL GTs for those samples, take worst (min).
        scores = score_fn(adv_act, [gts[int(i)] for i in idx])
        improved = scores < best_score[idx]
        # update best_x / best_score / best_gt
        for k, ii in enumerate(idx.tolist()):
            if improved[k]:
                best_x[ii] = adv_act[k].detach()
                best_score[ii] = scores[k].detach()
                best_gt[ii] = gt_idx

    # Phase 2: single-precision APGD on the worst GT, init from best_x.
    active = best_score > score_threshold
    if active.any():
        idx = active.nonzero(as_tuple=True)[0]
        for j in idx.tolist():
            gt_str = gts[j][int(best_gt[j])]
            stub = _score_fn_loss_wrapper(lvlm_loss_fn, gt_str)
            # init from current best perturbation
            x_init = best_x[j : j + 1]
            adv = apgd_attack(
                stub,
                x_init,
                torch.zeros(1, dtype=torch.long, device=x.device),
                eps=eps,
                n_iter=n_iter_high,
                loss_fn="ce",
                random_start=False,
            )
            score_after = score_fn(adv, [gts[j]])
            if score_after.item() < best_score[j].item():
                best_x[j] = adv[0].detach()
                best_score[j] = score_after[0].detach()

    return AttackResult(best_x, best_score, best_gt)


def targeted_attack(
    lvlm_loss_fn: Callable,
    x: torch.Tensor,
    target_caption: str,
    eps: float,
    n_iter: int = 10000,
    alpha: float = 1.0 / 255.0,
) -> torch.Tensor:
    """Stealthy targeted PGD: minimize the LVLM loss against `target_caption`.

    Per Sec. B.9, 10000 iterations are necessary to break LLaVA-CLIP at
    eps=2/255. Following Schlarmann & Hein (2023), the initial step size of
    APGD is set to eps; here we use plain PGD with a constant step alpha.
    """
    delta = torch.zeros_like(x).uniform_(-eps, eps)
    delta = (x + delta).clamp_(0.0, 1.0) - x
    delta.requires_grad_(True)

    for _ in range(n_iter):
        loss = lvlm_loss_fn(x + delta, target_caption).sum()
        # Targeted: *minimize* loss against the target caption -> step opposite to grad.
        grad = torch.autograd.grad(loss, delta)[0]
        delta = (delta.detach() - alpha * grad.sign()).clamp(-eps, eps)
        delta = (x + delta).clamp_(0.0, 1.0) - x
        delta.requires_grad_(True)

    return (x + delta).detach()
