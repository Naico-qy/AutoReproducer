"""PPO + Random Network Distillation trainer for Montezuma's Revenge (App. B.2).

Implements the standard PPO update with two value heads (one for extrinsic,
one for intrinsic reward) and an RND auxiliary loss that trains the predictor
network on a fraction `update_proportion` of each batch (Burda et al., 2018).

The Addendum specifies the implementation should follow:
    https://github.com/jcwleo/random-network-distillation-pytorch
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from . import aux_losses
from ..model.montezuma_net import (
    MontezumaPolicy,
    RNDPredictor,
    RNDTarget,
    rnd_intrinsic_reward,
)


@dataclass
class PPORNDConfig:
    total_steps: int = 50_000_000
    num_env: int = 128
    num_step: int = 128
    epoch: int = 4
    mini_batch: int = 4
    learning_rate: float = 1.0e-4
    clip_grad_norm: float = 0.5
    entropy: float = 0.001
    ppo_eps: float = 0.1
    gamma: float = 0.999
    int_gamma: float = 0.99
    lam: float = 0.95
    ext_coef: float = 2.0
    int_coef: float = 1.0
    update_proportion: float = 0.25
    use_gae: bool = True

    retention: str = "none"
    bc_loss_coef: float = 1.0
    ewc_lambda: float = 1.0e5
    ewc_apply_to_critic: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)


def compute_gae(
    rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float, lam: float
):
    """Standard GAE; rewards/values/dones shape (T, N)."""
    T, N = rewards.shape
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(N)
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - dones[t]
        next_value = values[t + 1] if t + 1 < T else values[-1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values[:T]
    return advantages, returns


class PPORNDTrainer:
    def __init__(
        self,
        policy: MontezumaPolicy,
        rnd_predictor: RNDPredictor,
        rnd_target: RNDTarget,
        teacher: Optional[MontezumaPolicy],
        config: PPORNDConfig,
        fisher: Optional[Dict[str, torch.Tensor]] = None,
        pretrained_params: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.policy = policy
        self.rnd_predictor = rnd_predictor
        self.rnd_target = rnd_target
        self.teacher = teacher
        if self.teacher is not None:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

        self.cfg = config
        self.optim = torch.optim.Adam(
            list(policy.parameters()) + list(rnd_predictor.parameters()),
            lr=config.learning_rate,
        )
        self.fisher = fisher or {}
        self.pretrained_params = pretrained_params or {}

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages_ext: torch.Tensor,
        advantages_int: torch.Tensor,
        returns_ext: torch.Tensor,
        returns_int: torch.Tensor,
        next_obs_for_rnd: torch.Tensor,
        bc_batch: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """One PPO+RND update over a rollout."""
        cfg = self.cfg
        # combined advantage following Burda et al. (2018)
        advantages = cfg.ext_coef * advantages_ext + cfg.int_coef * advantages_int
        adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- PPO loss
        logits, v_ext, v_int = self.policy(obs)
        log_pi = F.log_softmax(logits, dim=-1)
        log_pi_a = log_pi.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        ratio = torch.exp(log_pi_a - old_log_probs)
        surr1 = ratio * adv_norm
        surr2 = torch.clamp(ratio, 1.0 - cfg.ppo_eps, 1.0 + cfg.ppo_eps) * adv_norm
        pol_loss = -torch.min(surr1, surr2).mean()
        v_loss = (
            0.5 * ((v_ext - returns_ext).pow(2) + (v_int - returns_int).pow(2)).mean()
        )
        ent = -(log_pi.exp() * log_pi).sum(-1).mean()

        loss = pol_loss + v_loss - cfg.entropy * ent

        # ---- RND predictor loss (per-element, masked by update_proportion)
        with torch.no_grad():
            tgt = self.rnd_target(next_obs_for_rnd)
        pred = self.rnd_predictor(next_obs_for_rnd)
        rnd_err = (pred - tgt).pow(2).mean(-1)
        mask = (torch.rand_like(rnd_err) < cfg.update_proportion).float()
        denom = torch.clamp(mask.sum(), min=1.0)
        rnd_loss = (rnd_err * mask).sum() / denom
        loss = loss + rnd_loss

        # ---- Knowledge retention
        if cfg.retention == "behavioral_cloning" and bc_batch is not None:
            student_logits, _, _ = self.policy(bc_batch["obs"])
            with torch.no_grad():
                teacher_logits, _, _ = self.teacher(bc_batch["obs"])
            loss = loss + cfg.bc_loss_coef * aux_losses.bc_loss(
                student_logits, teacher_logits
            )
        elif cfg.retention == "ewc":
            named_params = (
                (n, p)
                for n, p in self.policy.named_parameters()
                if cfg.ewc_apply_to_critic or "value_" not in n
            )
            l_ewc = aux_losses.ewc_loss(
                (p for p in self.policy.parameters()),
                self.pretrained_params,
                self.fisher,
                named_params,
            )
            loss = loss + cfg.ewc_lambda * l_ewc

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.clip_grad_norm)
        self.optim.step()
        return {
            "loss": float(loss.item()),
            "policy_loss": float(pol_loss.item()),
            "value_loss": float(v_loss.item()),
            "rnd_loss": float(rnd_loss.item()),
            "entropy": float(ent.item()),
        }
