"""Score-network architecture used in NPSE / TSNPSE.

Implements the network described in Sharrock et al., ICML 2024,
"Sequential Neural Score Estimation: Likelihood-Free Inference with
Conditional Score Based Diffusion Models" (Appendix E.3.2).

Reference (verified via paper_search / arXiv):
    Greenberg, D. S., Nonnenmacher, M., & Macke, J. H. (2019).
    Automatic Posterior Transformation for Likelihood-Free Inference.
    ICML 2019. arXiv:1905.07488
    -- this is the SNPE-C baseline that NPSE/TSNPSE compares against.

The architecture (Appendix E.3.2 of the paper) is:
    * theta_t embedding network: 3-layer MLP, 256 hidden units, output dim
      max(30, 4*d) where d = dim(theta).
    * x embedding network: 3-layer MLP, 256 hidden units, output dim
      max(30, 4*p) where p = dim(x).
    * t sinusoidal embedding into 64 dimensions, following Vaswani et al. 2017.
    * Score network: concat([theta_emb, x_emb, t_emb]) -> 3-layer MLP, 256
      hidden units, output dim d.
    * SiLU activations everywhere.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


def _silu_mlp(
    in_dim: int, hidden: int, out_dim: int, n_layers: int = 3
) -> nn.Sequential:
    """Build a fully-connected MLP with `n_layers` hidden layers and SiLU activations.

    Per Appendix E.3.2: "3-layer fully-connected MLP with 256 hidden units in each
    layer". We interpret this as 3 hidden layers of width `hidden`, followed by a
    final linear projection to `out_dim`.
    """
    layers: list[nn.Module] = []
    last = in_dim
    for _ in range(n_layers):
        layers.append(nn.Linear(last, hidden))
        layers.append(nn.SiLU())
        last = hidden
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class MLPEmbedding(nn.Module):
    """3-layer MLP embedding for theta_t or x (Appendix E.3.2)."""

    def __init__(self, in_dim: int, hidden: int = 256, out_dim: Optional[int] = None):
        super().__init__()
        if out_dim is None:
            out_dim = max(30, 4 * in_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = _silu_mlp(in_dim, hidden, out_dim, n_layers=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for the diffusion time t in [0, 1].

    Implements exactly Appendix E.3.2:
        (t_emb)_i = sin( t / 10000^{(i-1)/31} )       if i <= 32
                  = cos( t / 10000^{((i-32)-1)/31} )  if i >  32
    Output dimension is 64.
    """

    def __init__(self, dim: int = 64):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalTimeEmbedding dim must be even, got {dim}")
        self.dim = dim
        half = dim // 2
        # Indices i = 1..half  →  exponent (i-1)/(half-1).
        denom = 10000.0 ** (torch.arange(half, dtype=torch.float32) / max(half - 1, 1))
        self.register_buffer("inv_freq", 1.0 / denom)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # `t` shape (B,) or (B, 1); produce (B, dim).
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        # Broadcast: (B,1) * (half,) -> (B, half)
        args = t * self.inv_freq.to(t.device).unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ScoreNetwork(nn.Module):
    """Time-varying conditional score network s_psi(theta_t, x, t).

    Approximates ``∇_theta log p_t(theta_t | x)`` (Eq. 7 of the paper).

    Architecture (Appendix E.3.2):
        theta_t  → MLPEmbedding(theta_dim)            → theta_emb (max(30,4d))
        x        → MLPEmbedding(x_dim)                → x_emb     (max(30,4p))
        t        → SinusoidalTimeEmbedding(64)        → t_emb     (64)
        concat([theta_emb, x_emb, t_emb])
                 → 3-layer MLP (256 hidden, SiLU)     → score in R^d
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        hidden: int = 256,
        time_emb_dim: int = 64,
        theta_emb_dim: Optional[int] = None,
        x_emb_dim: Optional[int] = None,
    ):
        super().__init__()
        self.theta_dim = theta_dim
        self.x_dim = x_dim

        self.theta_embedding = MLPEmbedding(
            theta_dim, hidden=hidden, out_dim=theta_emb_dim
        )
        self.x_embedding = MLPEmbedding(x_dim, hidden=hidden, out_dim=x_emb_dim)
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)

        joint_dim = (
            self.theta_embedding.out_dim + self.x_embedding.out_dim + time_emb_dim
        )
        # Final 3-layer MLP outputs the score in R^d.
        self.score_head = _silu_mlp(joint_dim, hidden, theta_dim, n_layers=3)

        # Standardisation buffers (Appendix E.3.3 "Standardization"):
        # we centre theta_t and x by subtracting an estimate of the mean and
        # dividing by the standard deviation in each dimension. The buffers
        # are populated by `set_normalization`.
        self.register_buffer("theta_mean", torch.zeros(theta_dim))
        self.register_buffer("theta_std", torch.ones(theta_dim))
        self.register_buffer("x_mean", torch.zeros(x_dim))
        self.register_buffer("x_std", torch.ones(x_dim))

    @torch.no_grad()
    def set_normalization(
        self,
        theta_mean: torch.Tensor,
        theta_std: torch.Tensor,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        eps: float = 1e-6,
    ) -> None:
        """Set per-dimension mean/std for input standardisation."""
        self.theta_mean.copy_(theta_mean)
        self.theta_std.copy_(theta_std.clamp_min(eps))
        self.x_mean.copy_(x_mean)
        self.x_std.copy_(x_std.clamp_min(eps))

    def _normalize_theta(self, theta: torch.Tensor) -> torch.Tensor:
        return (theta - self.theta_mean) / self.theta_std

    def _normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean) / self.x_std

    def forward(
        self,
        theta_t: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute s_psi(theta_t, x, t).

        Parameters
        ----------
        theta_t : (B, d) tensor of perturbed parameters.
        x       : (B, p) tensor of (observed / simulated) data.
        t       : (B,)    tensor of diffusion times in [0, 1].

        Returns
        -------
        score   : (B, d) tensor approximating ∇_theta log p_t(theta_t | x).
        """
        theta_n = self._normalize_theta(theta_t)
        x_n = self._normalize_x(x)

        theta_emb = self.theta_embedding(theta_n)
        x_emb = self.x_embedding(x_n)
        t_emb = self.time_embedding(t)

        joint = torch.cat([theta_emb, x_emb, t_emb], dim=-1)
        score_raw = self.score_head(joint)

        # The network operates on standardised theta. The raw score therefore
        # corresponds to ∇_{theta_norm} log p_t(theta_norm|x); via the chain
        # rule, the score wrt the un-standardised theta is divided by std.
        return score_raw / self.theta_std

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
