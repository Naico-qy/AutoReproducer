"""Forecasting models for "What Will My Model Forget?" (Jin & Ren, ICML 2024).

Three forecasters are implemented:

    threshold      §3.1  : ThresholdForecaster      (Eqn. 1)
    logit_based    §3.2  : LogitForecaster          (Eqn. 2 / 3)
                          FixedLogitForecaster      (no-train variant)
    repr_based     §3.3  : RepresentationForecaster (Eqn. 4 + freq-prior)
"""

from .architecture import BaseLM, wrap_with_lora, load_base_lm
from .encoder import Encoder, build_encoder
from .threshold import ThresholdForecaster
from .logit_forecaster import LogitForecaster, FixedLogitForecaster
from .repr_forecaster import RepresentationForecaster, frequency_prior
from .losses import margin_loss, binary_ce_loss

__all__ = [
    "BaseLM",
    "wrap_with_lora",
    "load_base_lm",
    "Encoder",
    "build_encoder",
    "ThresholdForecaster",
    "LogitForecaster",
    "FixedLogitForecaster",
    "RepresentationForecaster",
    "frequency_prior",
    "margin_loss",
    "binary_ce_loss",
]
