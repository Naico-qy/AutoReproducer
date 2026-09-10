"""SAC actor + Q-function for RoboticSequence (App. B.3 / Continual World).

Architecture:
    4-layer MLP, 256 neurons per hidden layer, LeakyReLU, LayerNorm after the
    first layer (Wołczyk et al., 2021). The actor has one Gaussian head per
    stage (one-hot stage ID picks the head); the critic has separate Q-heads
    per stage similarly. Per the paper, separate output heads outperformed
    appending the stage-ID to the state.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(
    in_dim: int, out_dim: int, hidden_dim: int, num_layers: int, layer_norm_first: bool
) -> nn.Sequential:
    layers = []
    layers.append(nn.Linear(in_dim, hidden_dim))
    if layer_norm_first:
        layers.append(nn.LayerNorm(hidden_dim))
    layers.append(nn.LeakyReLU())
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.LeakyReLU())
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """Multi-head Gaussian policy.

    Forward signature: `(obs, stage_idx) -> (mean, log_std)`.
    The forward also returns sampled actions and log-probs for SAC (a.k.a.
    `sample(...)`).
    """

    LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_stages: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        layer_norm_first: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_stages = num_stages
        # Shared trunk
        self.trunk = _mlp(obs_dim, hidden_dim, hidden_dim, num_layers, layer_norm_first)
        # Replace last layer of trunk with identity-style passthrough
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 2 * action_dim) for _ in range(num_stages)]
        )

    def _head(
        self, h: torch.Tensor, stage: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # `stage`: (B,) long tensor with values in [0, num_stages)
        out = torch.zeros(
            h.size(0), 2 * self.action_dim, device=h.device, dtype=h.dtype
        )
        for k, head in enumerate(self.heads):
            mask = stage == k
            if mask.any():
                out[mask] = head(h[mask])
        mean, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def forward(self, obs: torch.Tensor, stage: torch.Tensor):
        h = self.trunk(obs)
        return self._head(h, stage)

    def sample(self, obs: torch.Tensor, stage: torch.Tensor):
        """Reparameterised tanh-Gaussian sample with SAC's log-prob correction."""
        mean, log_std = self.forward(obs, stage)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        u = normal.rsample()
        a = torch.tanh(u)
        # log-prob with tanh squashing correction (Haarnoja et al., 2018)
        log_prob = normal.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return a, log_prob, mean

    def deterministic(self, obs: torch.Tensor, stage: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs, stage)
        return torch.tanh(mean)


class QFunction(nn.Module):
    """Twin Q-network (clipped double-Q from SAC) with stage-conditioned heads."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_stages: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        layer_norm_first: bool = True,
    ):
        super().__init__()
        self.q1 = _mlp(
            obs_dim + action_dim, num_stages, hidden_dim, num_layers, layer_norm_first
        )
        self.q2 = _mlp(
            obs_dim + action_dim, num_stages, hidden_dim, num_layers, layer_norm_first
        )
        self.num_stages = num_stages

    def _select(self, q: torch.Tensor, stage: torch.Tensor) -> torch.Tensor:
        return q.gather(1, stage.long().view(-1, 1)).squeeze(-1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, stage: torch.Tensor):
        x = torch.cat([obs, action], dim=-1)
        q1 = self._select(self.q1(x), stage)
        q2 = self._select(self.q2(x), stage)
        return q1, q2
