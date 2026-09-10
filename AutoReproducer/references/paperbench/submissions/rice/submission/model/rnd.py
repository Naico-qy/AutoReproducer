"""Random Network Distillation (Burda et al., 2018) — used by RICE Algorithm 2.

Per the paper §3.3 "Exploration with Random Network Distillation":
    R'(s_t, a_t) = R(s_t, a_t) + λ · ||f(s_{t+1}) - f̂(s_{t+1})||²

The target network ``f`` is randomly initialised and frozen. The predictor
network ``f̂`` is trained to regress to ``f(s)`` via MSE. The L2 distance
between them, after normalisation, is used as an intrinsic-reward bonus
that incentivises visits to under-explored states.

Verified citation (paper_search): Burda, Edwards, Storkey, Klimov.
"Exploration by Random Network Distillation", ICLR 2019. arXiv:1810.12894.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from utils.normalization import RunningMeanStd


def _mlp(in_dim: int, out_dim: int, hidden: Sequence[int]) -> nn.Sequential:
    layers, last = [], in_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        layers.append(nn.ReLU())
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class RNDModule(nn.Module):
    """Target + predictor heads with intrinsic-reward computation."""

    def __init__(
        self,
        obs_dim: int,
        feature_dim: int = 128,
        hidden: Sequence[int] = (128, 128),
        lr: float = 1e-4,
        normalize_obs: bool = True,
        normalize_reward: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.target = _mlp(obs_dim, feature_dim, hidden).to(device)
        self.predictor = _mlp(obs_dim, feature_dim, hidden).to(device)
        # Freeze target — Burda et al. 2018, §3.
        for p in self.target.parameters():
            p.requires_grad = False

        self.optim = Adam(self.predictor.parameters(), lr=lr)
        self.device = device
        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward
        self.obs_rms = RunningMeanStd(shape=(obs_dim,)) if normalize_obs else None
        self.rew_rms = RunningMeanStd(shape=()) if normalize_reward else None
        self._obs_clip = 5.0

    def _normalize(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_rms is None:
            return obs
        mean = torch.as_tensor(
            self.obs_rms.mean, dtype=torch.float32, device=obs.device
        )
        std = torch.as_tensor(
            np.sqrt(self.obs_rms.var) + 1e-8, dtype=torch.float32, device=obs.device
        )
        return torch.clamp((obs - mean) / std, -self._obs_clip, self._obs_clip)

    @torch.no_grad()
    def intrinsic_reward(self, next_obs_np: np.ndarray) -> np.ndarray:
        """Compute λ-free intrinsic reward — caller multiplies by λ."""
        if self.normalize_obs:
            self.obs_rms.update(next_obs_np)
        obs_t = torch.as_tensor(next_obs_np, dtype=torch.float32, device=self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        obs_t = self._normalize(obs_t)
        target_feat = self.target(obs_t)
        predict_feat = self.predictor(obs_t)
        per_sample = (predict_feat - target_feat).pow(2).mean(-1)
        rew = per_sample.cpu().numpy()
        if self.normalize_reward:
            self.rew_rms.update(rew)
            rew = rew / (np.sqrt(self.rew_rms.var) + 1e-8)
        return rew.squeeze()

    def update(self, obs_batch: np.ndarray) -> float:
        """One Adam step minimising MSE(predictor(s), target(s))."""
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        obs_t = self._normalize(obs_t)
        with torch.no_grad():
            target_feat = self.target(obs_t)
        predict_feat = self.predictor(obs_t)
        loss = (predict_feat - target_feat).pow(2).mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return float(loss.item())
