"""Soft Actor-Critic (Haarnoja et al. 2018) used for Meta-World.

Hyperparameters follow Table E.1:
  - Adam(0.9, 0.999), gamma=0.99, tau=0.005
  - alpha=0.2, autotuned, target update freq=1, policy update freq=2
  - actor lr = 1e-3, critic lr = 1e-3
  - replay buffer size = 1e6, batch size = 128
  - random actions before learning = 1e4, learn-start = 5e3

Notes per paper (Section 5.2 + Appendix E):
  - The CRL method is applied only to the actor; the critic is restarted
    at every task boundary.
  - The replay buffer is reset on task change.
  - The actor's output is the mean of a Gaussian; the log-std comes from
    a separate small network (CompoNetActor.log_std).
  - Action squashing follows the standard tanh-Gaussian SAC parameterisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SACConfig:
    """Hyperparameters for SAC (Table E.1)."""

    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    autotune_alpha: bool = True
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    alpha_lr: float = 1e-4
    batch_size: int = 128
    buffer_size: int = 1_000_000
    policy_update_freq: int = 2
    target_update_freq: int = 1
    learning_starts: int = 5_000
    random_actions: int = 10_000
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    noise_clip: float = 0.5  # not used by vanilla SAC, kept for completeness


class QNetwork(nn.Module):
    """Twin Q-network (per Haarnoja 2018)."""

    def __init__(
        self, obs_dim: int, act_dim: int, d_model: int = 256, num_layers: int = 3
    ) -> None:
        super().__init__()
        layers = [nn.Linear(obs_dim + act_dim, d_model), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(d_model, d_model), nn.ReLU(inplace=True)]
        layers += [nn.Linear(d_model, 1)]
        self.q1 = nn.Sequential(*layers)
        # second twin
        layers2 = [nn.Linear(obs_dim + act_dim, d_model), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers2 += [nn.Linear(d_model, d_model), nn.ReLU(inplace=True)]
        layers2 += [nn.Linear(d_model, 1)]
        self.q2 = nn.Sequential(*layers2)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class SAC:
    """SAC trainer (continuous actions)."""

    def __init__(
        self,
        actor: nn.Module,
        obs_dim: int,
        act_dim: int,
        cfg: Optional[SACConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.cfg = cfg or SACConfig()
        self.device = device
        self.actor = actor.to(device)
        self.qf = QNetwork(obs_dim, act_dim).to(device)
        self.qf_target = QNetwork(obs_dim, act_dim).to(device)
        self.qf_target.load_state_dict(self.qf.state_dict())
        for p in self.qf_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(
            [p for p in self.actor.parameters() if p.requires_grad],
            lr=self.cfg.actor_lr,
            betas=(0.9, 0.999),
        )
        self.qf_opt = torch.optim.Adam(
            self.qf.parameters(), lr=self.cfg.critic_lr, betas=(0.9, 0.999)
        )

        # Auto-tuning of alpha (Haarnoja et al. 2018, Sec 5).
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.tensor(
            np.log(self.cfg.alpha), requires_grad=True, device=device
        )
        self.alpha_opt = torch.optim.Adam(
            [self.log_alpha], lr=self.cfg.alpha_lr, betas=(0.9, 0.999)
        )
        self.act_dim = act_dim

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().detach()

    # -- action sampling -----------------------------------------------------
    def _sample_action(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample tanh-Gaussian action; return action, log_prob, mean."""
        mean, log_std = self.actor(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Tanh-correction (Haarnoja 2018, Appendix C, Eq. 21).
        log_prob = log_prob - torch.log(1.0 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return y_t, log_prob, torch.tanh(mean)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )
        if deterministic:
            mean, _ = self.actor(obs_t)
            return torch.tanh(mean).cpu().numpy()[0]
        a, _, _ = self._sample_action(obs_t)
        return a.cpu().numpy()[0]

    # -- update --------------------------------------------------------------
    def update(self, batch) -> dict:
        obs, act, rew, next_obs, done = batch
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        rew = torch.as_tensor(rew, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        # Critic update.
        with torch.no_grad():
            next_act, next_logp, _ = self._sample_action(next_obs)
            q1_t, q2_t = self.qf_target(next_obs, next_act)
            min_qt = torch.min(q1_t, q2_t) - self.alpha * next_logp
            target_q = rew + (1.0 - done) * self.cfg.gamma * min_qt
        q1, q2 = self.qf(obs, act)
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.qf_opt.zero_grad()
        q_loss.backward()
        self.qf_opt.step()

        # Actor update (delayed by policy_update_freq -- handled by trainer).
        a_pi, logp_pi, _ = self._sample_action(obs)
        q1_pi, q2_pi = self.qf(obs, a_pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * logp_pi - min_q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Alpha update.
        if self.cfg.autotune_alpha:
            alpha_loss = -(
                self.log_alpha * (logp_pi.detach() + self.target_entropy)
            ).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)

        # Soft update of target.
        self._soft_update(self.qf, self.qf_target, self.cfg.tau)

        return {
            "q_loss": float(q_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "alpha_loss": float(alpha_loss.item()),
        }

    @staticmethod
    def _soft_update(src: nn.Module, tgt: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for p, pt in zip(src.parameters(), tgt.parameters()):
                pt.data.mul_(1.0 - tau).add_(tau * p.data)

    # -- task transition -----------------------------------------------------
    def on_task_change(self, obs_dim: int, act_dim: int) -> None:
        """Reset the critic + replay buffer at task transitions (Section 5.2).

        For SAC, ``CompoNetActor`` should call ``add_new_task()`` separately.
        """
        self.qf = QNetwork(obs_dim, act_dim).to(self.device)
        self.qf_target = QNetwork(obs_dim, act_dim).to(self.device)
        self.qf_target.load_state_dict(self.qf.state_dict())
        for p in self.qf_target.parameters():
            p.requires_grad_(False)
        self.qf_opt = torch.optim.Adam(
            self.qf.parameters(), lr=self.cfg.critic_lr, betas=(0.9, 0.999)
        )
        # Reset the actor's optimizer to only the trainable params.
        self.actor_opt = torch.optim.Adam(
            [p for p in self.actor.parameters() if p.requires_grad],
            lr=self.cfg.actor_lr,
            betas=(0.9, 0.999),
        )
        # Reset alpha autotune.
        self.log_alpha = torch.tensor(
            np.log(self.cfg.alpha), requires_grad=True, device=self.device
        )
        self.alpha_opt = torch.optim.Adam(
            [self.log_alpha], lr=self.cfg.alpha_lr, betas=(0.9, 0.999)
        )
