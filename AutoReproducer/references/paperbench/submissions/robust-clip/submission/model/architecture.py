"""FARE model and loss.

Implements the central training objective of:
  Schlarmann, Singh, Croce, Hein. "Robust CLIP", ICML 2024, Eq. (3):

      L_FARE(theta; x) = max_{||z - x||_inf <= eps}
                          || phi_theta(z) - phi_Org(x) ||_2^2

  where phi_theta is the trainable CLIP vision encoder and phi_Org is the
  frozen original CLIP vision encoder. The class-token output is used
  (Sec. B.1).

Verified citation (CrossRef-style metadata; see ref_verify call in this
session):
  Mao, C. et al. "Understanding zero-shot adversarial robustness for
  large-scale models." ICLR 2023. — the supervised baseline TeCoA.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_loader import CLIPVisionWrapper


def fare_loss(
    phi_ft_z: torch.Tensor,
    phi_org_x: torch.Tensor,
    norm: str = "l2_squared",
) -> torch.Tensor:
    """FARE embedding loss (Eq. 3 of the paper).

    Parameters
    ----------
    phi_ft_z   : embeddings of the *adversarial* input under the trainable
                 fine-tuned encoder, shape (B, D).
    phi_org_x  : embeddings of the *clean* input under the frozen original
                 encoder, shape (B, D).
    norm       : "l2_squared" (default, paper main result) or "l1"
                 (ablation in Table 9). Both are mean-reduced over the batch.

    Returns
    -------
    Scalar loss tensor with grad flowing into phi_ft_z only.
    """
    # phi_org_x must be detached so the grad does not flow through it.
    phi_org_x = phi_org_x.detach()
    diff = phi_ft_z - phi_org_x
    if norm == "l2_squared":
        # Mean over batch of the squared L2 norm.
        return diff.pow(2).sum(dim=-1).mean()
    if norm == "l1":
        return diff.abs().sum(dim=-1).mean()
    raise ValueError(f"Unknown FARE loss norm: {norm}")


class FAREModel(nn.Module):
    """Bundles the (frozen) original and (trainable) fine-tune encoders.

    Use `.train_step(x_clean, x_adv)` to obtain (loss, phi_ft_z, phi_org_x)
    or run them separately for full control inside the inner-PGD loop.
    """

    def __init__(
        self,
        original: CLIPVisionWrapper,
        finetune: CLIPVisionWrapper,
        loss_norm: str = "l2_squared",
    ) -> None:
        super().__init__()
        self.original = original
        self.finetune = finetune
        self.loss_norm = loss_norm
        # Hard-freeze original; redundant with clip_loader but defensive.
        for p in self.original.parameters():
            p.requires_grad_(False)
        self.original.eval()

    @torch.no_grad()
    def encode_original(self, x: torch.Tensor) -> torch.Tensor:
        """phi_Org(x). Always frozen, no grad."""
        return self.original(x)

    def encode_finetune(self, x: torch.Tensor) -> torch.Tensor:
        """phi_FT(x). Grad-enabled when the model is in train mode."""
        return self.finetune(x)

    def forward(self, x_clean: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
        """Compute the FARE loss given a clean and adversarial input.

        The adversarial input z = x + delta should already be the maximizer
        from the inner PGD loop (see attacks/pgd.py). The original embedding
        is computed on the *clean* x, exactly per Eq. (3).
        """
        with torch.no_grad():
            phi_org = self.encode_original(x_clean)
        phi_ft = self.encode_finetune(x_adv)
        return fare_loss(phi_ft, phi_org, norm=self.loss_norm)

    def embeddings(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience: returns (phi_FT(x), phi_Org(x)) for analysis."""
        with torch.no_grad():
            phi_org = self.encode_original(x)
        phi_ft = self.encode_finetune(x)
        return phi_ft, phi_org

    def trainable_parameters(self):
        """Only the fine-tune encoder is trained — the AdamW optimizer
        receives parameters from this generator only (see train.py)."""
        return self.finetune.parameters()
