"""Stochastic-interpolant subpackage.

Implements Definition 3.1 of Albergo, Goldstein, Boffi, Ranganath,
Vanden-Eijnden, "Stochastic Interpolants with Data-Dependent Couplings",
ICML 2024.
"""

from .coefficients import (
    InterpolantCoefficients,
    LinearNoNoise,
    LinearWithNoise,
)
from .interpolant import StochasticInterpolant
from .losses import velocity_loss, score_loss
from .couplings import (
    Coupling,
    IndependentCoupling,
    InpaintingCoupling,
    SuperResolutionCoupling,
)
from .sampler import sample_ode_euler, sample_ode_dopri

__all__ = [
    "InterpolantCoefficients",
    "LinearNoNoise",
    "LinearWithNoise",
    "StochasticInterpolant",
    "velocity_loss",
    "score_loss",
    "Coupling",
    "IndependentCoupling",
    "InpaintingCoupling",
    "SuperResolutionCoupling",
    "sample_ode_euler",
    "sample_ode_dopri",
]
