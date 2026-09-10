"""Representation Descriptor (RD) for SEMA (Sec. 3.3, Eq. 2).

Per the addendum:
    "The representation descriptor is an AE with one encoder and decoder.
     The encoder and decoder include one linear layer followed by a non-linear
     activation. LeakyReLU is used in the application. The dimension of the
     latent space is 128 by default."

The RD is trained to minimise reconstruction error on all features fed to its
paired functional adapter (Eq. 2):

    L_RD = sum_{x in X^l_k} || x - g_phi(x) ||_2^2

It additionally maintains running statistics (mean, std) of the per-sample
reconstruction error so that, in future tasks, a z-score-based novelty
detector (Sec. 3.6) can decide whether to expand the model.
"""

from __future__ import annotations

import torch
from torch import nn


class RepresentationDescriptor(nn.Module):
    """Single-hidden-layer autoencoder with LeakyReLU non-linearities.

    Forward returns the reconstruction. Running statistics of the
    reconstruction error are exposed via :pyattr:`mu` and :pyattr:`sigma`.
    """

    def __init__(
        self,
        dim: int,
        latent_dim: int = 128,
        leaky_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(dim, latent_dim),
            nn.LeakyReLU(negative_slope=leaky_slope, inplace=False),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, dim),
            nn.LeakyReLU(negative_slope=leaky_slope, inplace=False),
        )

        # Running stats for z-score expansion signal (Sec. 3.6).
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("sigma", torch.ones(1))
        self.register_buffer("count", torch.zeros(1))

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    # ------------------------------------------------------- recon. error API
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample reconstruction error r_k^l = ||x - g(x)||_2^2.

        Accepts (B, D) or (B, N, D) tensors; in the latter case the error is
        averaged across the token dimension so that one scalar is produced
        per sample, mirroring the per-image novelty signal used in Fig. 4.
        """
        recon = self.forward(x)
        diff = (x - recon).pow(2)
        # Sum over feature dim, average over token dim if present.
        if diff.dim() == 3:
            err = diff.sum(dim=-1).mean(dim=-1)
        elif diff.dim() == 2:
            err = diff.sum(dim=-1)
        else:
            err = diff.flatten(1).sum(dim=-1)
        return err

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """L_RD (Eq. 2) reduced to a scalar mean for optimisation."""
        return self.reconstruction_error(x).mean()

    # ------------------------------------------------------ running statistics
    @torch.no_grad()
    def update_stats(self, x: torch.Tensor) -> None:
        """Update running mean / std of the reconstruction error.

        Uses Welford's online algorithm for numerical stability.
        """
        err = self.reconstruction_error(x.detach()).detach()
        for v in err.flatten().tolist():
            self.count += 1
            delta = v - self.mu.item()
            self.mu += delta / self.count
            delta2 = v - self.mu.item()
            self.sigma += delta * delta2  # store M2 in sigma temporarily

    @torch.no_grad()
    def finalise_stats(self) -> None:
        """Convert M2 -> standard deviation after `update_stats` accumulation.

        Should be called once after the descriptor finishes training on a task.
        """
        n = max(self.count.item(), 1.0)
        var = self.sigma.item() / n
        self.sigma.fill_(max(var, 1e-8) ** 0.5)

    @torch.no_grad()
    def z_score(self, x: torch.Tensor) -> torch.Tensor:
        """z = (r - mu) / sigma  -- batched (Sec. 3.6)."""
        err = self.reconstruction_error(x)
        sigma = self.sigma.clamp_min(1e-6)
        return (err - self.mu) / sigma
