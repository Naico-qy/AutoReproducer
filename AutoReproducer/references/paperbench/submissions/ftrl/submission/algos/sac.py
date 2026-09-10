"""Soft Actor-Critic trainer for RoboticSequence (App. B.3).

Follows Spinning Up's SAC formulation, the implementation referenced by the
Addendum: https://spinningup.openai.com/en/latest/algorithms/sac.html

Key details (App. B.3):
    * twin Q-networks, target networks, Polyak update tau=0.005
    * automatic entropy temperature tuning (Haarnoja et al., 2018b)
    * separate output head per stage, picked by stage-id
    * MLP 4 layers x 256 with LeakyReLU, LayerNorm after first layer
    * batch_size = 128, lr = 1e-3
    * critic NEVER receives the retention loss (App. C.5)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from . import aux_losses
from ..model.sac_net import GaussianActor, QFunction


@dataclass
class SACConfig:
    total_steps: int = 4_000_000
    batch_size: int = 128
    buffer_size: int = 100_000
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 1.0e-3
    auto_entropy: bool = True
    target_entropy: Optional[float] = None  # defaults to -action_dim
    log_likelihood_every_steps: int = 50_000  # Addendum (Figure 8)

    retention: str = "none"
    bc_actor_coef: float = 1.0
    bc_critic_coef: float = 0.0
    ewc_actor_coef: float = 100.0
    ewc_critic_coef: float = 0.0
    bc_buffer_size: int = 10000
    em_buffer_size: int = 10000
    extras: Dict[str, Any] = field(default_factory=dict)


class SACTrainer:
    def __init__(
        self,
        actor: GaussianActor,
        critic: QFunction,
        teacher_actor: Optional[GaussianActor],
        config: SACConfig,
        device: str = "cpu",
        fisher: Optional[Dict[str, torch.Tensor]] = None,
        pretrained_params: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.teacher_actor = teacher_actor
        if self.teacher_actor is not None:
            self.teacher_actor = self.teacher_actor.to(device)
            for p in self.teacher_actor.parameters():
                p.requires_grad = False

        self.cfg = config
        self.device = device

        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=config.learning_rate
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=config.learning_rate
        )

        if config.auto_entropy:
            target_entropy = config.target_entropy
            if target_entropy is None:
                target_entropy = -float(self.actor.action_dim)
            self.target_entropy = target_entropy
            self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
            self.alpha_optim = torch.optim.Adam(
                [self.log_alpha], lr=config.learning_rate
            )
        else:
            self.log_alpha = torch.tensor(np.log(0.2), device=device)
            self.alpha_optim = None

        self.fisher = fisher or {}
        self.pretrained_params = pretrained_params or {}
        self._train_step = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ---- target net Polyak update -----------------------------------------

    def _polyak(self):
        with torch.no_grad():
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

    # ---- one update step --------------------------------------------------

    def update(
        self,
        batch: Dict[str, torch.Tensor],
        bc_batch: Optional[Dict[str, torch.Tensor]] = None,
    ):
        cfg = self.cfg
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]
        stage = batch["stage"]

        # ---- Critic update ------------------------------------------------
        with torch.no_grad():
            next_a, next_logp, _ = self.actor.sample(next_obs, stage)
            tq1, tq2 = self.target_critic(next_obs, next_a, stage)
            tq = torch.min(tq1, tq2) - self.alpha.detach() * next_logp.squeeze(-1)
            target = reward + (1.0 - done) * cfg.gamma * tq

        q1, q2 = self.critic(obs, action, stage)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optim.step()

        # ---- Actor update -------------------------------------------------
        new_a, logp, _ = self.actor.sample(obs, stage)
        q1_pi, q2_pi = self.critic(obs, new_a, stage)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * logp.squeeze(-1) - q_pi).mean()

        # ---- Knowledge retention auxiliary actor losses (NEVER on critic)
        if cfg.retention == "behavioral_cloning" and bc_batch is not None:
            s_obs = bc_batch["obs"]
            s_stage = bc_batch["stage"]
            s_mean, s_logstd = self.actor(s_obs, s_stage)
            with torch.no_grad():
                t_mean, t_logstd = self.teacher_actor(s_obs, s_stage)
            actor_loss = actor_loss + cfg.bc_actor_coef * aux_losses.bc_loss_gaussian(
                s_mean, s_logstd, t_mean, t_logstd
            )
        elif cfg.retention == "ewc":
            named_params = ((n, p) for n, p in self.actor.named_parameters())
            l_ewc = aux_losses.ewc_loss(
                (p for p in self.actor.parameters()),
                self.pretrained_params,
                self.fisher,
                named_params,
            )
            actor_loss = actor_loss + cfg.ewc_actor_coef * l_ewc

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optim.step()

        # ---- Entropy temperature update -----------------------------------
        if self.alpha_optim is not None:
            alpha_loss = -(
                self.log_alpha * (logp.squeeze(-1).detach() + self.target_entropy)
            ).mean()
            self.alpha_optim.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optim.step()

        self._polyak()
        self._train_step += 1

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "alpha": float(self.alpha.detach().cpu()),
        }

    # ---- analysis (Figure 8) ----------------------------------------------

    def log_expert_likelihood(self, expert_batch: Dict[str, torch.Tensor]) -> float:
        """Log-likelihood of expert (s, a*) under the current policy.

        Used in the FPC analysis (App. F / Figure 8). Sampling cadence = every
        50 000 training steps per the Addendum.
        """
        if self._train_step % self.cfg.log_likelihood_every_steps != 0:
            return float("nan")

        obs = expert_batch["obs"].to(self.device)
        a = expert_batch["action"].to(self.device)
        stage = expert_batch["stage"].to(self.device)
        with torch.no_grad():
            mean, log_std = self.actor(obs, stage)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            # log-prob of pre-tanh action: invert tanh (clamped)
            a_clamped = a.clamp(-0.999999, 0.999999)
            u = 0.5 * (a_clamped.log1p() - (-a_clamped).log1p())
            logp = normal.log_prob(u) - torch.log(1 - a_clamped.pow(2) + 1e-6)
            logp = logp.sum(dim=-1)
        return float(logp.mean().cpu())
