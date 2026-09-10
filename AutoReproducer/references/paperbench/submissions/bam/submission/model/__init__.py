"""Target-distribution and neural-network models used in the BaM experiments."""

from .architecture import VAE, Encoder, Decoder, vae_log_prior_likelihood_score
from .targets import (
    GaussianTarget,
    SinhArcsinhTarget,
    PosteriorDBTarget,
    VAEPosteriorTarget,
    make_random_gaussian_target,
)

__all__ = [
    "VAE",
    "Encoder",
    "Decoder",
    "vae_log_prior_likelihood_score",
    "GaussianTarget",
    "SinhArcsinhTarget",
    "PosteriorDBTarget",
    "VAEPosteriorTarget",
    "make_random_gaussian_target",
]
