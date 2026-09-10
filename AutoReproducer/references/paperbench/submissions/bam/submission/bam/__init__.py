"""
Batch and Match (BaM): Black-box Variational Inference with a Score-based Divergence
Cai, Modi, Pillaud-Vivien, Margossian, Gower, Blei, Saul. ICML 2024.

This package implements the BaM algorithm, the GSM baseline (Modi et al. 2023, NeurIPS,
verified via citation-grounded retrieval
in Section 5 of the paper.
"""

from .bam import BaM, BaMState, bam_update, low_rank_bam_update
from .gsm import GSM
from .advi import ADVI
from .gradient_methods import ScoreVI, FisherVI
from .divergences import (
    score_based_divergence,
    fisher_divergence,
    forward_kl_gaussian,
    reverse_kl_gaussian,
)

__all__ = [
    "BaM",
    "BaMState",
    "bam_update",
    "low_rank_bam_update",
    "GSM",
    "ADVI",
    "ScoreVI",
    "FisherVI",
    "score_based_divergence",
    "fisher_divergence",
    "forward_kl_gaussian",
    "reverse_kl_gaussian",
]
