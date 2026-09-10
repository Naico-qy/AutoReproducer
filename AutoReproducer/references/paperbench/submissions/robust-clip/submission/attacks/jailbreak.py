"""Visual jailbreak attack on LVLMs (Sec. 4.4 / Table 7).

Per addendum:
  * Implementation adapted from Qi et al. (2023):
    https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models
  * 5000 iterations, alpha = 1/255.
  * No momentum (plain sign-PGD).
  * Single seed image: clean.jpeg from the Qi et al. repo.
  * Universal targeted attack: targets are harmful "derogatory_corpus.csv"
    strings; loss is the mean negative log-likelihood across that corpus.

We test eps in {16/255, 32/255, 64/255} (Table 7).
"""

from __future__ import annotations

from typing import Callable, List

import torch


def jailbreak_universal_attack(
    lvlm_loss_fn: Callable[[torch.Tensor, str], torch.Tensor],
    seed_image: torch.Tensor,  # (1, 3, H, W) in [0, 1] — Qi et al. clean.jpeg
    harmful_targets: List[str],  # derogatory_corpus.csv strings
    eps: float,
    alpha: float = 1.0 / 255.0,
    n_iter: int = 5000,
) -> torch.Tensor:
    """Run the universal targeted PGD attack from Qi et al. (2023).

    Returns a single perturbed image (1, 3, H, W) that is then queried with
    each of the 40 harmful prompts in `manual_harmful_instructions.csv` to
    measure the attack success rate.
    """
    assert seed_image.dim() == 4 and seed_image.shape[0] == 1
    delta = torch.empty_like(seed_image).uniform_(-eps, eps)
    delta = (seed_image + delta).clamp_(0.0, 1.0) - seed_image
    delta.requires_grad_(True)

    for it in range(n_iter):
        # Sample a single target string per iteration (mirrors Qi et al. impl).
        target = harmful_targets[it % len(harmful_targets)]
        x_perturbed = seed_image + delta
        loss = lvlm_loss_fn(x_perturbed, target).sum()
        grad = torch.autograd.grad(loss, delta)[0]
        # Targeted -> step against gradient direction.
        delta = (delta.detach() - alpha * grad.sign()).clamp_(-eps, eps)
        delta = (seed_image + delta).clamp_(0.0, 1.0) - seed_image
        delta.requires_grad_(True)

    return (seed_image + delta).detach()
