"""Coreset selection baselines reimplemented for LBCS comparisons.

Per addendum, the authors reimplemented these from the original repos:
  - Uniform sampling
  - EL2N         (Paul et al., NeurIPS 2021)
  - GraNd        (Paul et al., NeurIPS 2021)
  - Influential  (Yang et al., ICLR 2023)
  - Moderate     (Xia et al., ICLR 2023b)
  - CCS          (Zheng et al., ICLR 2023)
  - Probabilistic (Zhou et al., ICML 2022) — bilevel
"""

from .scoring import (
    uniform_select,
    el2n_select,
    grand_select,
    moderate_select,
    ccs_select,
    influential_select,
    probabilistic_select,
)

__all__ = [
    "uniform_select",
    "el2n_select",
    "grand_select",
    "moderate_select",
    "ccs_select",
    "influential_select",
    "probabilistic_select",
]
