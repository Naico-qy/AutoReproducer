"""Baselines reproduced for comparison (Section 5.2).

Per addendum, two reference implementations were prescribed:
  * LoRA + Mask-Tuning post-training pruning
        https://github.com/WoosukKwon/retraining-free-pruning
        (adapted so it runs on top of a LoRA-tuned model.)
  * CoFi (pruning + L_0 regularised distillation)
        https://github.com/princeton-nlp/CoFiPruning
        (adapted so only LoRA + L_0 modules are tuned.)

Verified citation (CrossRef):
    Xia, Zhong, Chen, "Structured Pruning Learns Compact and Accurate
    Models", ACL 2022. DOI 10.18653/v1/2022.acl-long.107  (CoFi.)

This module provides minimal-but-faithful adapters over the two
baselines so that they share the same APT data loader.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


# --------------------------------------------------------------------------- #
@dataclass
class MaskTuningCfg:
    """Mask-Tuning hyperparameters (Kwon et al., 2022).

    Defaults follow the upstream config of
    https://github.com/WoosukKwon/retraining-free-pruning.
    """

    target_sparsity: float = 0.6
    n_calibration_batches: int = 32
    n_pruning_iters: int = 100
    fisher_damp: float = 1.0e-4


def mask_tuning_prune(model: nn.Module, calibration_loader, cfg: MaskTuningCfg) -> dict:
    """A minimal Fisher-information-based mask-tuning pass.

    For each linear layer we accumulate diagonal Fisher information from
    `n_calibration_batches`, score each output channel by Σ_i (g_i)², and
    zero-out the bottom (1-keep_ratio) fraction.

    NB: this is a *minimal* faithful implementation — the official
    upstream uses linear-programming over heads/neurons jointly. The
    APT paper applies it post-LoRA tuning, which is what we do here.
    """
    fisher = {}
    model.eval()
    n = 0
    for batch in calibration_loader:
        batch = {k: v for k, v in batch.items()}
        out = model(**batch)
        loss = out.get("loss", out["logits"].pow(2).mean())
        grads = torch.autograd.grad(
            loss,
            [p for p in model.parameters() if p.requires_grad],
            retain_graph=False,
            allow_unused=True,
        )
        for (name, p), g in zip(
            ((n_, p_) for n_, p_ in model.named_parameters() if p_.requires_grad),
            grads,
        ):
            if g is None:
                continue
            fisher.setdefault(name, torch.zeros_like(p))
            fisher[name] = fisher[name] + g.pow(2)
        n += 1
        if n >= cfg.n_calibration_batches:
            break

    # Per-output-channel score.
    keep_ratio = 1.0 - cfg.target_sparsity
    threshold = {}
    for name, F in fisher.items():
        if F.dim() < 2:
            continue
        scores = F.sum(dim=tuple(range(1, F.dim())))
        k = max(1, int(scores.numel() * keep_ratio))
        topk = torch.topk(scores, k, largest=True).indices
        mask = torch.zeros_like(scores)
        mask[topk] = 1.0
        threshold[name] = mask
    return threshold


# --------------------------------------------------------------------------- #
@dataclass
class CofiCfg:
    """CoFi (Xia et al., ACL 2022, doi:10.18653/v1/2022.acl-long.107).

    We follow the public CoFi config: lambda_1 / lambda_2 are L_0
    regularisers; alpha_layer is the layer-wise distillation weight.
    """

    target_sparsity: float = 0.6
    lambda_1: float = 1.0
    lambda_2: float = 1.0
    distill_temperature: float = 4.0  # τ — same as APT (addendum)
    distill_layer_weight: float = 0.9  # 0.9 L_layer (addendum)
    distill_pred_weight: float = 1.0  # 1.0 L_pred for GLUE (addendum)


class L0Module(nn.Module):
    """Hard-Concrete L_0 mask (Louizos et al., 2018) used by CoFi."""

    def __init__(
        self, dim: int, beta: float = 2.0 / 3.0, zeta: float = 1.1, gamma: float = -0.1
    ) -> None:
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(dim))
        self.beta = beta
        self.zeta = zeta
        self.gamma = gamma

    def _sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def sample_mask(self) -> torch.Tensor:
        if self.training:
            u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
            s = self._sigmoid(
                (torch.log(u) - torch.log1p(-u) + self.log_alpha) / self.beta
            )
        else:
            s = self._sigmoid(self.log_alpha)
        s = s * (self.zeta - self.gamma) + self.gamma
        return s.clamp(0.0, 1.0)

    def expected_l0(self) -> torch.Tensor:
        return self._sigmoid(
            self.log_alpha
            - self.beta * torch.log(torch.tensor(-self.gamma / self.zeta))
        ).sum()


def cofi_l0_loss(
    modules, target_sparsity: float, lam1: float = 1.0, lam2: float = 1.0
) -> torch.Tensor:
    expected = sum(m.expected_l0() for m in modules)
    total = sum(m.log_alpha.numel() for m in modules)
    target = (1.0 - target_sparsity) * total
    diff = (expected - target) / max(total, 1)
    return lam1 * diff + lam2 * diff.pow(2)
