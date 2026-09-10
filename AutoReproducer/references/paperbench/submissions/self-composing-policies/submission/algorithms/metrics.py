"""CRL metrics from Section 5.1 of the paper.

Definitions:
  - p_i(t): success rate (0 or 1) of task i at timestep t.
  - Performance:  P(t) = (1/N) sum_i p_i(t)
  - AUC_i        = (1/Delta) integral_{(i-1)Delta}^{i Delta} p_i(t) dt
  - AUC^b_i      = (1/Delta) integral_0^{Delta}  p_i^b(t) dt   (baseline)
  - FTr_i        = (AUC_i - AUC^b_i) / (1 - AUC^b_i)
  - RT           = (1/N) sum_{i=2..N} max_{j<i} FTr(j, i)        (Eq. 3)
  - Forgetting:  F_i = p_i(i*Delta) - p_i(T)                      (Sec F.2)
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def average_performance(p_at_T: Sequence[float]) -> float:
    """Final average performance P(T) = mean of per-task success rates at T."""
    return float(np.mean(p_at_T))


def _auc(curve: Sequence[float]) -> float:
    """Trapezoid AUC on [0,1] of a discrete curve sampled at uniform t."""
    arr = np.asarray(curve, dtype=np.float64)
    if arr.size < 2:
        return float(arr.mean()) if arr.size else 0.0
    return float(np.trapz(arr, dx=1.0 / (arr.size - 1)))


def forward_transfer(
    method_curve: Sequence[float], baseline_curve: Sequence[float]
) -> float:
    """Forward transfer FTr_i for a single task (Equation 2)."""
    auc_m = _auc(method_curve)
    auc_b = _auc(baseline_curve)
    if 1.0 - auc_b <= 1e-12:
        return 0.0
    return (auc_m - auc_b) / (1.0 - auc_b)


def reference_transfer(transfer_matrix: np.ndarray) -> float:
    """Reference forward Transfer (Equation 3).

    Args:
        transfer_matrix : (N, N) matrix where M[j, i] = FTr(j -> i).
            Diagonal corresponds to fine-tuning the same task; for RT we
            take the max over j<i excluding j==i.

    Returns:
        RT scalar.
    """
    M = np.asarray(transfer_matrix, dtype=np.float64)
    N = M.shape[0]
    if N < 2:
        return 0.0
    s = 0.0
    for i in range(1, N):
        # j < i, exclude diagonal? Paper says "max_{j<i} FTr(j, i)" without
        # excluding the diagonal -- but since j<i, diagonal is automatically
        # excluded from the range.
        s += float(M[:i, i].max())
    return s / (N - 1)


def forgetting(p_end_of_task: float, p_end_of_sequence: float) -> float:
    """Forgetting F_i = p_i(i*Delta) - p_i(T) (Equation 4 in Sec F.2)."""
    return float(p_end_of_task - p_end_of_sequence)


def success_rate_curve(
    returns: Sequence[float], success_threshold: float
) -> List[float]:
    """Convert per-episode returns to a success-rate curve.

    Each entry of `returns` is a single-episode return; we mark success=1
    if return >= threshold, else 0.  This implements the Section 5.1 /
    Appendix D.4 definition for SpaceInvaders / Freeway.
    """
    return [1.0 if r >= success_threshold else 0.0 for r in returns]
