"""Misc utilities for FOA: ECE computation, memory measurement, seeding."""

from __future__ import annotations

import os
import random
from typing import List, Tuple

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    """Set torch / numpy / python RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def expected_calibration_error(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Standard ECE with equal-width confidence bins (paper uses 15 bins).

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    confidences = np.asarray(confidences).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if confidences.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = float(confidences[mask].mean())
        avg_acc = float((predictions[mask] == labels[mask]).mean())
        ece += (mask.sum() / confidences.size) * abs(avg_conf - avg_acc)
    return float(100.0 * ece)


def measure_peak_memory_mb() -> float:
    """Reset and report peak CUDA memory in MB (per addendum.md)."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated()) / (1024 * 1024)


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
