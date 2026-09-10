"""Proximal Policy Optimization (Schulman et al. 2017) for ALE sequences.

Hyperparameters follow Table E.2:
  - Adam(0.9, 0.999), lr=2.5e-4 with linear annealing.
  - gamma=0.99, GAE lambda=0.95, clip=0.2, value coef=0.5, entropy=0.01.
  - max grad norm=0.5, advantage normalisation, value loss clipping.
  - 8 parallel envs, 128 rollout steps, 4 mini-batches of 256, 4 update epochs.

Implementation derived from CleanRL [Huang et al. JMLR 2022], cited by the
paper (Appendix E).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    num_envs: int = 8
    num_steps: int = 128
    update_epochs: int = 4
    num_minibatches: int = 4
    norm_adv: bool = True
    clip_vloss: bool = True


class CriticNet(nn.Module):
    """A small value head; the encoder is shared with the actor."""

    def __init__(self, d_enc: int, d_model: int = 512) -> None:
        super().__init__()
        self.value_head = nn.Linear(d_enc, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.value_head(h).squeeze(-1)


class PPO:
    """PPO (clip) trainer for discrete actions (ALE)."""

    def __init__(
        self,
        actor: nn.Module,
        encoder: nn.Module,
        n_actions: int,
        d_enc: int = 512,
        cfg: Optional[PPOConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.cfg = cfg or PPOConfig()
        self.device = device
        self.actor = actor.to(device)
        self.encoder = encoder.to(device)
        self.critic = CriticNet(d_enc).to(device)
        self.n_actions = n_actions
        params = (
            list(self.actor.parameters())
            + list(self.encoder.parameters())
            + list(self.critic.parameters())
        )
        self.opt = torch.optim.Adam(
            [p for p in params if p.requires_grad],
            lr=self.cfg.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-5,
        )
        self._task_total_updates: int = 0
        self._task_update_counter: int = 0

    # -- action sampling -----------------------------------------------------
    @torch.no_grad()
    def select_action(self, obs: torch.Tensor):
        """Return action, log_prob, value for one obs tensor."""
        h = self.encoder(obs)
        probs = (
            self.actor.componet(h) if hasattr(self.actor, "componet") else self.actor(h)
        )
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        return action, dist.log_prob(action), self.critic(h)

    # -- update --------------------------------------------------------------
    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
    ) -> dict:
        """Single PPO update over `update_epochs * num_minibatches` minibatches.

        All inputs are flattened to (T*N, ...) tensors.
        """
        cfg = self.cfg
        b_inds = np.arange(obs.shape[0])
        mb_size = obs.shape[0] // cfg.num_minibatches

        if cfg.anneal_lr and self._task_total_updates > 0:
            frac = 1.0 - (self._task_update_counter / self._task_total_updates)
            new_lr = cfg.learning_rate * max(frac, 0.0)
            for g in self.opt.param_groups:
                g["lr"] = new_lr

        last = {"loss": 0.0}
        for _ in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for s in range(0, obs.shape[0], mb_size):
                idx = b_inds[s : s + mb_size]
                mb_obs = obs[idx]
                mb_act = actions[idx]
                mb_lp_old = logprobs[idx]
                mb_adv = advantages[idx]
                mb_ret = returns[idx]
                mb_val_old = values[idx]

                h = self.encoder(mb_obs)
                probs = (
                    self.actor.componet(h)
                    if hasattr(self.actor, "componet")
                    else self.actor(h)
                )
                dist = torch.distributions.Categorical(probs=probs.clamp_min(1e-8))
                new_lp = dist.log_prob(mb_act)
                entropy = dist.entropy()
                new_val = self.critic(h)

                logratio = new_lp - mb_lp_old
                ratio = logratio.exp()

                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Clipped surrogate objective.
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                # Value loss (clipped).
                if cfg.clip_vloss:
                    v_unclipped = (new_val - mb_ret) ** 2
                    v_clipped = mb_val_old + torch.clamp(
                        new_val - mb_val_old, -cfg.clip_coef, cfg.clip_coef
                    )
                    v_clipped = (v_clipped - mb_ret) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_val - mb_ret) ** 2).mean()

                ent_loss = entropy.mean()
                loss = pg_loss + cfg.value_coef * v_loss - cfg.entropy_coef * ent_loss

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [
                        p
                        for p in (
                            list(self.actor.parameters())
                            + list(self.encoder.parameters())
                            + list(self.critic.parameters())
                        )
                        if p.requires_grad
                    ],
                    cfg.max_grad_norm,
                )
                self.opt.step()
                last = {
                    "pg_loss": float(pg_loss.item()),
                    "v_loss": float(v_loss.item()),
                    "ent_loss": float(ent_loss.item()),
                    "loss": float(loss.item()),
                }
        self._task_update_counter += 1
        return last

    # -- GAE -----------------------------------------------------------------
    @staticmethod
    def compute_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> torch.Tensor:
        """Generalized Advantage Estimation (Schulman et al. 2016)."""
        T = rewards.shape[0]
        adv = torch.zeros_like(rewards)
        last_gae = torch.zeros(rewards.shape[1], device=rewards.device)
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = last_value
                next_nonterminal = 1.0 - dones[t]
            else:
                next_val = values[t + 1]
                next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_val * next_nonterminal - values[t]
            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + values
        return adv, returns

    # -- task transition -----------------------------------------------------
    def on_task_change(self) -> None:
        """Reinitialise critic at task boundary (Section 5.2)."""
        self.critic = CriticNet(self.critic.value_head.in_features).to(self.device)
        # Re-initialise actor head optimiser parameter group.
        params = (
            list(self.actor.parameters())
            + list(self.encoder.parameters())
            + list(self.critic.parameters())
        )
        self.opt = torch.optim.Adam(
            [p for p in params if p.requires_grad],
            lr=self.cfg.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-5,
        )
        self._task_update_counter = 0
