"""Salience scoring — Section 4.2 / 4.3 of APT (Zhao et al., ICML 2024).

Three salience flavours are implemented:

  1) `parameter_salience`        — Eq. (3): |W ⊙ ∂L/∂W|
                                   For tunable weights (W_A, W_B in the
                                   APT adapter) where gradients are
                                   reachable.

  2) `outlier_aware_salience`    — Eq. (4)+(5): activation-based score
                                   for *frozen* parameters whose gradients
                                   are unavailable. Augmented with the
                                   square-root of activation kurtosis to
                                   keep outlier blocks (§4.2).

  3) `adapter_salience`          — Eq. (3) summed over all entries of W_B
                                   (or equivalently W_A; the paper notes
                                   in footnote 3 they're equal). Used
                                   to rank APT adapters when growing
                                   ranks (§4.3).

A small `EMASalience` helper realises the addendum's
    Ŝ_t(m) = 0.85 Ŝ_{t-1}(m) + 0.15 S_hat(m)
exponential moving-average smoother.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch


# --------------------------------------------------------------------------- #
def parameter_salience(
    weight: torch.Tensor, grad: Optional[torch.Tensor]
) -> torch.Tensor:
    """Eq. (3): S(W_{i,j}) = |W_{i,j} · ∂L/∂W_{i,j}|."""
    if grad is None:
        return torch.zeros_like(weight)
    return (weight * grad).abs()


# --------------------------------------------------------------------------- #
def _kurtosis(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-block kurtosis along the last dim (Pearson definition).

    Section 4.2: 'Representing the density of the outlier in a
    distribution, the more outliers there are, the bigger the kurtosis.'
    Returns a positive scalar per block.
    """
    mean = x.mean(dim=-1, keepdim=True)
    centred = x - mean
    var = centred.pow(2).mean(dim=-1, keepdim=True).clamp_min(eps)
    fourth = centred.pow(4).mean(dim=-1)
    return fourth / var.squeeze(-1).pow(2).clamp_min(eps)


def outlier_aware_salience(
    activation_abs_sum: torch.Tensor,
    grad_abs_sum: torch.Tensor,
    use_kurtosis: bool = True,
) -> torch.Tensor:
    """Eqs. (4)+(5) of the paper.

    Inputs are already summed-over-batch absolute values (memory-saving
    trick described in §4.2: 'we compress the activation and gradients by
    summing along batches before production').

    Args:
        activation_abs_sum : (d,)  Σ |H_{j,i}|   over batch & sequence
        grad_abs_sum       : (d,)  Σ |∂L/∂H_{j,i}|
        use_kurtosis       : whether to add the √Kurt(O) outlier term

    Returns:
        Tensor of shape (d,) — outlier-aware salience score per block.
    """
    base = activation_abs_sum * grad_abs_sum  # element-wise product
    if not use_kurtosis:
        return base
    # Treat each block (entry j) as a 1-element distribution after batch
    # summing.  Following the spirit of Eq. (5) we approximate per-block
    # kurtosis by Kurt of the activation flattened across hidden dim.
    if activation_abs_sum.numel() > 1:
        k = _kurtosis(activation_abs_sum.unsqueeze(0)).squeeze(0)
        return base + k.sqrt().clamp_min(0.0)
    return base


# --------------------------------------------------------------------------- #
def adapter_salience(adapter) -> float:
    """I(H_apt) = Σ_{i,j} |W_B_{i,j} · ∂L/∂W_B_{i,j}|, §4.3."""
    if adapter.W_B.grad is None:
        return 0.0
    return parameter_salience(adapter.W_B.data, adapter.W_B.grad).sum().item()


# --------------------------------------------------------------------------- #
class EMASalience:
    """Exponential moving-average over salience scores.

    Implements addendum:
        S_bar^{(t)}(m) ← 0.85 · S_bar^{(t-1)}(m) + 0.15 · S_hat(m)
    """

    def __init__(self, decay: float = 0.85) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must lie in [0, 1)")
        self.decay = decay
        self._state: Dict[str, torch.Tensor] = {}

    def update(self, name: str, score: torch.Tensor) -> torch.Tensor:
        score = score.detach().float()
        prev = self._state.get(name)
        if prev is None or prev.shape != score.shape:
            new = score.clone()
        else:
            new = self.decay * prev + (1.0 - self.decay) * score
        self._state[name] = new
        return new

    def get(self, name: str) -> Optional[torch.Tensor]:
        return self._state.get(name)

    def keys(self) -> Iterable[str]:
        return self._state.keys()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self._state.items()}

    def load_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        self._state = {k: v.clone() for k, v in sd.items()}
