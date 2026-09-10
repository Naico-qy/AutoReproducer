"""LCA Alignment Loss (Algorithm 1, Appendix E.2).

This is the auxiliary soft-label loss used in §4.3.2 of the paper to
augment the standard cross-entropy when training a linear probe on
frozen backbone features. The total loss is

    L = lambda_weight * L_CE  +  L_soft_lca

where L_soft_lca uses the *inverted* LCA distance matrix (1 - M_LCA) as
"soft labels", thereby encouraging the model to assign secondary
likelihood to classes that are semantically close to the ground truth.

Hyperparameters from the paper (Appendix E.2):
    lambda_weight = 0.03
    temperature   = 25
    alignment     = 'CE' (cross-entropy soft loss)

Linear interpolation in weight space (paper §4.3.2):
    W_interp = alpha * W_ce + (1 - alpha) * W_ce+soft         (Wortsman 2022)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LCAAlignmentLoss(nn.Module):
    """Faithful reproduction of the algorithm in Appendix E.2.

    Args:
        lca_matrix: (K, K) tensor of post-processed LCA distances in [0, 1].
            This is the OUTPUT of `process_lca_matrix` (i.e. M_LCA in the
            paper). The loss internally builds reverse_LCA = 1 - M_LCA.
            Sanity check (addendum): reverse_LCA must have ones on the
            diagonal (since D_LCA(i, i) = 0 implies M_LCA[i, i] = 0).
        alignment_mode: 'CE' (default) or 'BCE'; matches Algorithm 1 in the
            paper.
        lambda_weight: scalar weight on the standard CE loss; paper default
            0.03 (Appendix E.2).
    """

    def __init__(
        self,
        lca_matrix: torch.Tensor,
        alignment_mode: str = "CE",
        lambda_weight: float = 0.03,
    ) -> None:
        super().__init__()
        if alignment_mode not in ("CE", "BCE"):
            raise ValueError(f"alignment_mode must be CE or BCE, got {alignment_mode}")
        # reverse_LCA_matrix = 1 - LCA_matrix  (Algorithm 1, line 2)
        reverse = 1.0 - lca_matrix
        # Sanity check from the addendum: diagonal of reversed matrix == 1.
        diag = reverse.diag()
        assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5), (
            "Inverted LCA matrix must have ones on the diagonal (addendum)."
        )
        self.register_buffer("reverse_lca", reverse)
        self.alignment_mode = alignment_mode
        self.lambda_weight = lambda_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the total LCA-aligned loss.

        Args:
            logits: (B, K) class logits (pre-softmax).
            targets: (B,) integer class indices.
        """
        # Standard cross-entropy: L_CE = -sum(one_hot * log(probs))
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        standard_loss = -(one_hot * log_probs).sum(dim=1)  # (B,)

        # Soft loss using the inverted LCA matrix as multi-label targets.
        soft_targets = self.reverse_lca[targets]  # (B, K)
        if self.alignment_mode == "BCE":
            criterion = nn.BCEWithLogitsLoss(reduction="none")
            soft_loss = criterion(logits, soft_targets).mean(dim=1)  # (B,)
        else:  # 'CE'
            # L_soft = - mean over classes of soft_targets * log(probs)
            soft_loss = -(soft_targets * log_probs).mean(dim=1)  # (B,)

        total = self.lambda_weight * standard_loss + soft_loss
        return total.mean()


def weight_interpolate(
    state_ce: dict,
    state_soft: dict,
    alpha: float,
) -> dict:
    """Linear interpolation in weight space (paper §4.3.2):

        W_interp = alpha * W_ce + (1 - alpha) * W_ce+soft.

    Args:
        state_ce: state_dict of the CE-only model.
        state_soft: state_dict of the CE+soft model.
        alpha: interpolation coefficient in [0, 1].

    Returns:
        A new state_dict whose tensors are alpha-mix of the two inputs.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    out: dict = {}
    for k, v_ce in state_ce.items():
        if k not in state_soft:
            out[k] = v_ce
            continue
        v_s = state_soft[k]
        if torch.is_tensor(v_ce) and torch.is_tensor(v_s):
            out[k] = alpha * v_ce.float() + (1.0 - alpha) * v_s.float()
        else:
            out[k] = v_ce
    return out
