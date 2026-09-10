"""Asynchronous PPO trainer for NetHack (paper App. B.1).

This is a self-contained PyTorch APPO that mirrors the recipe in
sample-factory (Petrenko et al., 2020), the implementation pinned by the
Addendum. We keep one actor process running rollouts and a learner that
applies the V-trace-free PPO update with the APPO-specific *separate* policy
and baseline clip ranges (Table 1 / App. B.1):

    appo_clip_policy   = 0.1
    appo_clip_baseline = 1.0

Auxiliary losses (BC/KS/EWC) are added per `--retention`. `freeze_encoders=True`
freezes the three encoders for the entire fine-tuning, mirroring App. B.1
("To improve the stability of the models we froze the encoders during the
course of the training").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from . import aux_losses


@dataclass
class APPOConfig:
    total_steps: int = 500_000_000
    unroll_length: int = 32
    batch_size: int = 128
    discounting: float = 0.999999
    appo_clip_policy: float = 0.1
    appo_clip_baseline: float = 1.0
    baseline_cost: float = 1.0
    entropy_cost: float = 0.001
    grad_norm_clipping: float = 4.0
    adam_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    adam_eps: float = 1.0e-7
    reward_clip: float = 10.0
    reward_scale: float = 1.0

    retention: str = "none"  # one of {none, behavioral_cloning, kickstarting, ewc}
    bc_loss_coef: float = 2.0
    bc_decay: float = 1.0
    ks_loss_coef: float = 0.5
    ks_decay: float = 0.99998
    ewc_lambda: float = 2.0e6
    ewc_apply_to_critic: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)


def _logits_to_log_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits, dim=-1)


def appo_policy_loss(
    new_logits: torch.Tensor,
    old_log_probs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    """PPO-style clipped surrogate, but with APPO's separate policy clip range."""
    log_pi = _logits_to_log_probs(new_logits)
    new_log_pi_a = log_pi.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    ratio = torch.exp(new_log_pi_a - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
    return -torch.min(surr1, surr2).mean()


def appo_baseline_loss(
    new_baseline: torch.Tensor,
    old_baseline: torch.Tensor,
    returns: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    """APPO uses a separate (much larger) clip on the value function (clip=1.0)."""
    v_clipped = old_baseline + torch.clamp(new_baseline - old_baseline, -clip, clip)
    loss_unclipped = (new_baseline - returns).pow(2)
    loss_clipped = (v_clipped - returns).pow(2)
    return torch.max(loss_unclipped, loss_clipped).mean()


def entropy_bonus(logits: torch.Tensor) -> torch.Tensor:
    log_pi = _logits_to_log_probs(logits)
    pi = log_pi.exp()
    return -(pi * log_pi).sum(dim=-1).mean()


class APPOTrainer:
    """Minimal APPO learner that mirrors Sample Factory's update step."""

    def __init__(
        self,
        model: torch.nn.Module,
        teacher: Optional[torch.nn.Module],
        config: APPOConfig,
        fisher: Optional[Dict[str, torch.Tensor]] = None,
        pretrained_params: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.model = model
        self.teacher = teacher
        if self.teacher is not None:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()
        self.config = config

        # AdamW with paper hyperparameters
        self.optim = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=config.adam_learning_rate,
            weight_decay=config.weight_decay,
            eps=config.adam_eps,
            betas=(0.9, 0.999),
        )

        self.fisher = fisher or {}
        self.pretrained_params = pretrained_params or {}
        self._train_step = 0

    # ----- one APPO update --------------------------------------------------

    def update(
        self,
        rollout: Dict[str, torch.Tensor],
        bc_batch: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Apply one APPO + retention update.

        Required rollout keys: chars, colors, blstats, message, action,
        old_log_prob, advantage, return, baseline, done.
        """
        cfg = self.config
        obs = {k: rollout[k] for k in ("chars", "colors", "blstats", "message")}
        new_logits, new_baseline, _ = self.model(obs, done_mask=rollout.get("not_done"))

        pol_loss = appo_policy_loss(
            new_logits,
            rollout["old_log_prob"],
            rollout["action"],
            rollout["advantage"],
            cfg.appo_clip_policy,
        )
        bas_loss = appo_baseline_loss(
            new_baseline, rollout["baseline"], rollout["return"], cfg.appo_clip_baseline
        )
        ent = entropy_bonus(new_logits)
        loss = pol_loss + cfg.baseline_cost * bas_loss - cfg.entropy_cost * ent

        # ---- Knowledge retention auxiliary losses --------------------------
        if cfg.retention == "behavioral_cloning" and bc_batch is not None:
            student_logits, _, _ = self.model(
                {k: bc_batch[k] for k in ("chars", "colors", "blstats", "message")}
            )
            with torch.no_grad():
                teacher_logits, _, _ = self.teacher(
                    {k: bc_batch[k] for k in ("chars", "colors", "blstats", "message")}
                )
            scale = cfg.bc_loss_coef * (cfg.bc_decay**self._train_step)
            loss = loss + scale * aux_losses.bc_loss(student_logits, teacher_logits)

        elif cfg.retention == "kickstarting":
            with torch.no_grad():
                teacher_logits, _, _ = self.teacher(
                    obs, done_mask=rollout.get("not_done")
                )
            scale = aux_losses.ks_decay_schedule(
                cfg.ks_loss_coef, cfg.ks_decay, self._train_step
            )
            loss = loss + scale * aux_losses.ks_loss(new_logits, teacher_logits)

        elif cfg.retention == "ewc":
            named_params = (
                (n, p)
                for n, p in self.model.named_parameters()
                if cfg.ewc_apply_to_critic or "baseline_head" not in n
            )
            l_ewc = aux_losses.ewc_loss(
                (p for p in self.model.parameters()),
                self.pretrained_params,
                self.fisher,
                named_params,
            )
            loss = loss + cfg.ewc_lambda * l_ewc

        # ---- Optimisation step --------------------------------------------
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in self.model.parameters() if p.requires_grad),
            cfg.grad_norm_clipping,
        )
        self.optim.step()
        self._train_step += 1
        return {
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(pol_loss.detach().cpu()),
            "baseline_loss": float(bas_loss.detach().cpu()),
            "entropy": float(ent.detach().cpu()),
        }
