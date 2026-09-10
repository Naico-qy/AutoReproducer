"""
Convolutional VAE for the deep-generative-model experiment of Section 5.3.

Architecture follows the addendum exactly:

* Encoder
    Conv(3 -> c_hid,        k=3, s=2)  -> 16x16
    Conv(c_hid -> c_hid,    k=3, s=1)  -> 16x16
    Conv(c_hid -> 2c_hid,   k=3, s=2)  -> 8x8
    Conv(2c_hid -> 2c_hid,  k=3, s=1)  -> 8x8
    Conv(2c_hid -> 2c_hid,  k=3, s=2)  -> 4x4
    Flatten -> Dense(latent_dim)
* Decoder
    Dense(latent_dim) -> reshape to [B, 2c_hid, 4, 4]
    ConvT(2c_hid -> 2c_hid, k=3, s=2)  -> 8x8
    Conv(2c_hid  -> 2c_hid, k=3, s=1)  -> 8x8
    ConvT(2c_hid -> c_hid,  k=3, s=2)  -> 16x16
    Conv(c_hid   -> c_hid,  k=3, s=1)  -> 16x16
    ConvT(c_hid  -> 3,      k=3, s=2)  -> 32x32
    tanh activation                    -> outputs in [-1, 1]
* Activations: GELU in all hidden layers, tanh on final decoder layer.
* No dropout, no normalization, no pooling (downsampling via stride=2 conv).
* latent_dim = 256, sigma^2 = 0.1 in the likelihood (paper).
* Adam optimizer with linear warmup 0 -> 1e-4 over 100 steps, then linear
  decay to 1e-5 over 500 training batches.

The training-time loss is a single-MC-sample negative ELBO (mc_sim = 1, per
addendum).  The score function used by BaM at inference time is the gradient
of  log p(z') p(x' | z')  with respect to z', evaluated by autograd through
the trained decoder.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:  # pragma: no cover -- optional GPU dependency
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# PyTorch model definitions
# ---------------------------------------------------------------------------


if HAS_TORCH:

    class Encoder(nn.Module):
        def __init__(self, c_hid: int = 32, latent_dim: int = 256):
            super().__init__()
            self.act = nn.GELU()
            self.c1 = nn.Conv2d(3, c_hid, kernel_size=3, stride=2, padding=1)
            self.c2 = nn.Conv2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1)
            self.c3 = nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, stride=2, padding=1)
            self.c4 = nn.Conv2d(
                2 * c_hid, 2 * c_hid, kernel_size=3, stride=1, padding=1
            )
            self.c5 = nn.Conv2d(
                2 * c_hid, 2 * c_hid, kernel_size=3, stride=2, padding=1
            )
            # 4 * 4 * (2*c_hid) flattened.
            self.fc_mu = nn.Linear(4 * 4 * 2 * c_hid, latent_dim)
            self.fc_logvar = nn.Linear(4 * 4 * 2 * c_hid, latent_dim)

        def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            h = self.act(self.c1(x))
            h = self.act(self.c2(h))
            h = self.act(self.c3(h))
            h = self.act(self.c4(h))
            h = self.act(self.c5(h))
            h = h.flatten(start_dim=1)
            return self.fc_mu(h), self.fc_logvar(h)

    class Decoder(nn.Module):
        def __init__(self, c_hid: int = 32, latent_dim: int = 256):
            super().__init__()
            self.c_hid = c_hid
            self.act = nn.GELU()
            self.fc = nn.Linear(latent_dim, 4 * 4 * 2 * c_hid)
            self.t1 = nn.ConvTranspose2d(
                2 * c_hid,
                2 * c_hid,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
            self.c1 = nn.Conv2d(
                2 * c_hid, 2 * c_hid, kernel_size=3, stride=1, padding=1
            )
            self.t2 = nn.ConvTranspose2d(
                2 * c_hid, c_hid, kernel_size=3, stride=2, padding=1, output_padding=1
            )
            self.c2 = nn.Conv2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1)
            self.t3 = nn.ConvTranspose2d(
                c_hid, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            )

        def forward(self, z: "torch.Tensor") -> "torch.Tensor":
            h = self.fc(z).view(-1, 2 * self.c_hid, 4, 4)
            h = self.act(self.t1(h))
            h = self.act(self.c1(h))
            h = self.act(self.t2(h))
            h = self.act(self.c2(h))
            return torch.tanh(self.t3(h))

    class VAE(nn.Module):
        """Convolutional VAE for CIFAR-10 (Section 5.3).

        Likelihood: x | z ~ N(decoder(z, theta), sigma^2 I) with sigma^2 = 0.1.
        Prior:      z ~ N(0, I_{latent_dim}).
        """

        def __init__(self, c_hid: int = 32, latent_dim: int = 256, sigma2: float = 0.1):
            super().__init__()
            self.encoder = Encoder(c_hid=c_hid, latent_dim=latent_dim)
            self.decoder = Decoder(c_hid=c_hid, latent_dim=latent_dim)
            self.latent_dim = latent_dim
            self.sigma2 = sigma2

        def reparameterize(
            self, mu: "torch.Tensor", logvar: "torch.Tensor"
        ) -> "torch.Tensor":
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(
            self, x: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            mu, logvar = self.encoder(x)
            z = self.reparameterize(mu, logvar)
            x_hat = self.decoder(z)
            return x_hat, mu, logvar

        # ------------------------------------------------------------------
        def neg_elbo(self, x: "torch.Tensor", mc_sim: int = 1) -> "torch.Tensor":
            """Single-sample negative ELBO (default mc_sim=1, per addendum)."""
            mu, logvar = self.encoder(x)
            B, D = mu.shape
            losses = []
            for _ in range(mc_sim):
                z = self.reparameterize(mu, logvar)
                x_hat = self.decoder(z)
                recon = 0.5 * ((x - x_hat) ** 2).flatten(1).sum(-1) / self.sigma2
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)
                losses.append(recon + kl)
            return torch.stack(losses, dim=0).mean()

else:  # numpy-only stubs that document the interface

    class Encoder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not available; install torch to use VAE.")

    class Decoder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not available; install torch to use VAE.")

    class VAE:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not available; install torch to use VAE.")


# ---------------------------------------------------------------------------
# Score function for posterior z' | x'
# ---------------------------------------------------------------------------


def vae_log_prior_likelihood_score(decoder, x_obs: np.ndarray, sigma2: float = 0.1):
    """Build a NumPy-callable score function s(z) = grad log p(z, x_obs).

    The unnormalized log target is

        log p(z) + log p(x_obs | z)
            = -0.5 ||z||^2 - (1/(2 sigma^2)) ||x_obs - Omega(z, theta)||^2 + const.

    We close over the trained PyTorch decoder and use autograd to compute
    grad_z of this expression.  The returned callable accepts either a
    single (D,) vector or a batch (B, D) and returns the same shape.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for VAE score function.")
    device = next(decoder.parameters()).device
    x_t = torch.as_tensor(x_obs, dtype=torch.float32, device=device).view(1, 3, 32, 32)

    def score(z: np.ndarray) -> np.ndarray:
        single = z.ndim == 1
        z_in = z[None, :] if single else z
        z_t = torch.as_tensor(z_in, dtype=torch.float32, device=device).requires_grad_(
            True
        )
        x_hat = decoder(z_t)
        # log prior + log likelihood, summed over batch.
        log_prior = -0.5 * (z_t**2).sum(dim=1)
        recon = -0.5 * ((x_t - x_hat) ** 2).flatten(1).sum(-1) / sigma2
        log_target = (log_prior + recon).sum()
        grad = torch.autograd.grad(log_target, z_t)[0].detach().cpu().numpy()
        return grad[0] if single else grad

    return score
