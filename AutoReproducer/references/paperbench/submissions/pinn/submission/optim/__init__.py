"""Optimizer module: NysNewton-CG (NNCG) and helper utilities.

Implements Algorithms 4–7 from Appendix E.2 of Rathore et al. (2024).
"""

from .nncg import NysNewtonCG
from .nystrom import randomized_nystrom_approximation, NystromPreconditioner
from .pcg import nystrom_pcg

__all__ = [
    "NysNewtonCG",
    "randomized_nystrom_approximation",
    "NystromPreconditioner",
    "nystrom_pcg",
]
