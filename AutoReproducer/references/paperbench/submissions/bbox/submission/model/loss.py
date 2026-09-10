"""
Loss functions for BBox-Adapter.

Implements the **ranking-based Noise Contrastive Estimation (NCE)** loss
of Section 3.2 / Eq. (2) of the paper, plus the ℓ2 spectral-norm
regulariser of Eq. (3) (per the addendum item 1: "spectral
normalization is implemented as ℓ2 regularization of the energies").

The MLM baseline used in the ablation Table 5 is also provided.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Eq. (2):
#   l(theta) = - E_{p_data}[ g_theta(x_+) - log sum_k exp g_theta(x_k) ]
#
# Eq. (3) (with α-regularised energies):
#   ∇ l(θ) = ∇ { -E[g(x_+)] + α E[g(x_+)^2]
#                + E_{p_θ}[g(x_-)] + α E[g(x_-)^2] }
#
# In practice we form, **per question**:
#   * one positive energy           e_+     = g_θ(x, y_+)
#   * K negative energies          e_-^k    = g_θ(x, y_-^k)   (k = 1..K)
# and compute the InfoNCE-style cross entropy:
#   loss_nce = -e_+ + log( exp(e_+) + Σ_k exp(e_-^k) )
# plus the α regulariser on every energy.
# ----------------------------------------------------------------------
def ranking_nce_loss(
    pos_energy: torch.Tensor,  # (B,)
    neg_energy: torch.Tensor,  # (B, K)
    alpha: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """
    Ranking-based NCE loss of Eq. (2)-(3).

    Args
    ----
    pos_energy : (B,)
        g_theta(x_i, y_i^+) for the positive sample of each question.
    neg_energy : (B, K)
        g_theta(x_i, y_{i,k}^-) for K negatives per question (drawn from
        p_theta — the adapted inference, see §3.4).
    alpha : float
        Coefficient of the ℓ2 spectral-norm term (Eq. 3, addendum).

    Returns
    -------
    dict with keys
        'loss'   : scalar — sum of NCE term and ℓ2 regulariser
        'nce'    : scalar — InfoNCE term only (for logging)
        'reg'    : scalar — α * E[g^2]
        'acc@1'  : scalar — fraction of questions where the positive
                            outranks every negative (paper's de-facto
                            quality metric for the adapter on a held-out
                            subset).
    """
    if pos_energy.dim() != 1:
        pos_energy = pos_energy.view(-1)
    if neg_energy.dim() == 1:
        neg_energy = neg_energy.unsqueeze(0)
    assert pos_energy.shape[0] == neg_energy.shape[0], "batch mismatch"

    # logits per question: [pos, neg_1, ..., neg_K]
    logits = torch.cat([pos_energy.unsqueeze(1), neg_energy], dim=1)  # (B, 1+K)
    # The positive is always at position 0 by construction.
    target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

    nce = F.cross_entropy(logits, target, reduction="mean")

    # ℓ2 regulariser α E[g^2] on all energies (positives + negatives).
    sq_pos = pos_energy.pow(2).mean()
    sq_neg = neg_energy.pow(2).mean()
    reg = alpha * (sq_pos + sq_neg)

    loss = nce + reg

    with torch.no_grad():
        acc1 = (logits.argmax(dim=1) == 0).float().mean()

    return {"loss": loss, "nce": nce.detach(), "reg": reg.detach(), "acc@1": acc1}


# ----------------------------------------------------------------------
# Ablation: Masked Language Modelling baseline (Table 5).
# We re-use the encoder's MLM head if available; otherwise we fall back
# to a linear projection over the vocabulary.  This corresponds to the
# "MLM" rows of Table 5 in the paper.
# ----------------------------------------------------------------------
def mlm_loss(
    logits: torch.Tensor,  # (B, T, V)
    labels: torch.Tensor,  # (B, T) with -100 for unmasked positions
) -> torch.Tensor:
    """
    Standard MLM cross-entropy (ignore_index = -100).  Matches the
    "MLM" baseline reported in Table 5 of the paper.
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )


# ----------------------------------------------------------------------
# Convenience: pack a list of positive / negative text pairs and run the
# adapter forward pass before computing the NCE loss.  Used by train.py.
# ----------------------------------------------------------------------
def compute_nce_batch_loss(
    adapter,
    questions: List[str],
    positives: List[str],
    negatives: List[List[str]],
    alpha: float = 0.01,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute Eq. (2)+(3) on a list of (question, y_+, [y_-]) tuples.

    Args
    ----
    adapter   : BBoxAdapter
    questions : list of B questions
    positives : list of B positive answers (one per question)
    negatives : list of B lists, each of length K — negative answers
    alpha     : ℓ2 spectral-norm coefficient
    """
    B = len(questions)
    K = len(negatives[0])
    assert all(len(neg) == K for neg in negatives), "negatives must be rectangular"

    # Build flat lists for batched forward pass.
    flat_q: List[str] = []
    flat_a: List[str] = []
    for q, y_pos, y_neg_list in zip(questions, positives, negatives):
        flat_q.append(q)
        flat_a.append(y_pos)
        for y_neg in y_neg_list:
            flat_q.append(q)
            flat_a.append(y_neg)

    enc = adapter.encode_pair(flat_q, flat_a, device=device)
    energies = adapter(**enc)  # (B*(1+K),)
    energies = energies.view(B, 1 + K)
    pos_e = energies[:, 0]
    neg_e = energies[:, 1:]
    return ranking_nce_loss(pos_e, neg_e, alpha=alpha)
