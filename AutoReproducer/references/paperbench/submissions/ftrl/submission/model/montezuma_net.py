"""Montezuma's Revenge PPO + RND networks (App. B.2).

Mirrors the architecture used by https://github.com/jcwleo/random-network-distillation-pytorch
(the implementation pinned in the Addendum and Table 2). It consists of:

    * `MontezumaPolicy` - shared CNN trunk producing a feature vector that
      feeds two separate value heads (extrinsic and intrinsic) and a discrete
      policy over 18 Atari actions.
    * `RNDTarget`        - randomly initialised CNN that maps an 84x84
      grayscale frame to a 512-dim feature; weights are frozen.
    * `RNDPredictor`     - trained CNN that tries to match `RNDTarget`'s
      output. Its squared error is the intrinsic reward.

All networks expect a stack of 4 frames (state_stack_size=4).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_out(size: int, kernel: int, stride: int, padding: int = 0) -> int:
    return (size + 2 * padding - kernel) // stride + 1


class _NatureCNN(nn.Module):
    """The Nature DQN CNN trunk used by Burda et al. (2018)."""

    def __init__(self, in_channels: int = 4, feature_dim: int = 512):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        # 84 -> 20 -> 9 -> 7
        self.fc = nn.Linear(64 * 7 * 7, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.flatten(1)
        return F.relu(self.fc(x))


class MontezumaPolicy(nn.Module):
    """PPO actor-critic with separate extrinsic and intrinsic value heads."""

    NUM_ACTIONS = 18

    def __init__(
        self,
        in_channels: int = 4,
        feature_dim: int = 512,
        num_actions: int = NUM_ACTIONS,
    ):
        super().__init__()
        self.cnn = _NatureCNN(in_channels=in_channels, feature_dim=feature_dim)
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, 448),
            nn.ReLU(),
            nn.Linear(448, 448),
            nn.ReLU(),
            nn.Linear(448, num_actions),
        )
        self.value_ext = nn.Sequential(
            nn.Linear(feature_dim, 448),
            nn.ReLU(),
            nn.Linear(448, 448),
            nn.ReLU(),
            nn.Linear(448, 1),
        )
        self.value_int = nn.Sequential(
            nn.Linear(feature_dim, 448),
            nn.ReLU(),
            nn.Linear(448, 448),
            nn.ReLU(),
            nn.Linear(448, 1),
        )
        self.num_actions = num_actions

    def forward(self, obs: torch.Tensor):
        f = self.cnn(obs)
        logits = self.policy(f)
        v_ext = self.value_ext(f).squeeze(-1)
        v_int = self.value_int(f).squeeze(-1)
        return logits, v_ext, v_int

    def actor_parameters(self):
        for name, p in self.named_parameters():
            if "value_" not in name:
                yield p


class RNDTarget(nn.Module):
    """Random target network, kept frozen (Burda et al., 2018)."""

    def __init__(self, in_channels: int = 1, feature_dim: int = 512):
        super().__init__()
        self.cnn = _NatureCNN(in_channels=in_channels, feature_dim=feature_dim)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.cnn(obs)


class RNDPredictor(nn.Module):
    """Trained predictor network; outputs are matched against the target."""

    def __init__(self, in_channels: int = 1, feature_dim: int = 512):
        super().__init__()
        self.cnn = _NatureCNN(in_channels=in_channels, feature_dim=feature_dim)
        # The original paper uses an additional MLP head on top of the trunk.
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.cnn(obs))


def rnd_intrinsic_reward(
    predictor: RNDPredictor, target: RNDTarget, obs: torch.Tensor
) -> torch.Tensor:
    """`r_int(s) = ||predictor(s) - target(s)||^2`. Burda et al. (2018), Eq. 1."""
    with torch.no_grad():
        t = target(obs)
    p = predictor(obs)
    return (p - t).pow(2).sum(dim=-1)
