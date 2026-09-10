"""Mask-Network training (Algorithm 1 of the paper).

Per §3.3 of Cheng et al. (2024), we re-formulate the StateMask objective as

    J(θ) = max  η(π̄)

where π̄ is the perturbed agent (target action mixed with the random action
through the binary mask). To prevent the trivial solution "never blind", we
add a per-step blinding bonus:

    R'(s_t, a_t) = R(s_t, a_t) + α · a^m_t

This recasts mask training as a vanilla PPO problem (which the paper exploits
to gain a 16.8% training-time speedup over StateMask). The implementation
below is a self-contained PPO loop that consumes mask-net actions, executes
the *target* policy underneath, applies the action-mixing operator from
Eq. (1), and returns the trained ``MaskNet``.

Reference: Cheng et al., "RICE", ICML 2024 — Algorithm 1.
Verified baseline (paper_search): Cheng et al., "StateMask", NeurIPS 2023.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .architecture import ActorCritic, MaskNet


@dataclass
class MaskTrainConfig:
    total_timesteps: int = 500_000
    alpha: float = 0.001  # blinding bonus
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    max_grad_norm: float = 0.5
    device: str = "cpu"


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    n = len(rewards)
    for t in reversed(range(n)):
        next_value = last_value if t == n - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


class MaskNetworkTrainer:
    """Trains the mask network via PPO with the §3.3 reward augmentation."""

    def __init__(self, env, target_policy: ActorCritic, cfg: MaskTrainConfig):
        self.env = env
        self.target_policy = target_policy.eval()  # frozen
        for p in self.target_policy.parameters():
            p.requires_grad = False

        obs_space = env.observation_space
        obs_dim = int(np.prod(obs_space.shape))
        self.mask_net = MaskNet(obs_dim).to(cfg.device)
        self.optim = Adam(self.mask_net.parameters(), lr=cfg.learning_rate)
        self.cfg = cfg
        self.device = cfg.device

        # action-space metadata for the random-action sampler in Eq. (1).
        self.action_space = env.action_space
        self.action_dim = (
            self.action_space.shape[0]
            if hasattr(self.action_space, "shape") and len(self.action_space.shape) > 0
            else self.action_space.n
        )
        self.continuous_actions = hasattr(self.action_space, "high")

    # ---------------------------------------------------------------- helpers
    def _sample_random_action(self):
        """``a_random`` from Eq. (1): a uniform sample from the action space."""
        return self.action_space.sample()

    def _action_mixing(self, target_action: np.ndarray, mask_bit: int):
        """Implements Eq. (1):  a_t ⊙ a^m_t."""
        if mask_bit == 0:
            return target_action
        return self._sample_random_action()

    @torch.no_grad()
    def _query_target(self, obs_np: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(
            obs_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        action, _, _ = self.target_policy.act(obs, deterministic=False)
        return action.squeeze(0).cpu().numpy()

    # ----------------------------------------------------------------- train
    def train(self):
        cfg = self.cfg
        rollout = _Rollout(cfg.n_steps, self.env.observation_space.shape)
        timestep = 0
        obs, _ = self.env.reset()

        while timestep < cfg.total_timesteps:
            rollout.reset()
            for step in range(cfg.n_steps):
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                mask_dist, value = self.mask_net(obs_t)
                mask_bit = int(mask_dist.sample().item())
                log_prob = mask_dist.log_prob(
                    torch.as_tensor([mask_bit], device=self.device)
                ).item()

                target_action = self._query_target(obs)
                actual_action = self._action_mixing(target_action, mask_bit)

                next_obs, reward, term, trunc, _ = self.env.step(actual_action)
                done = bool(term or trunc)
                # §3.3: R'(s,a) = R(s,a) + α · a^m
                augmented_reward = float(reward) + cfg.alpha * float(mask_bit)

                rollout.add(
                    obs, mask_bit, augmented_reward, done, value.item(), log_prob
                )

                obs = next_obs
                timestep += 1
                if done:
                    obs, _ = self.env.reset()

            # bootstrap last value
            with torch.no_grad():
                last_obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                _, last_value = self.mask_net(last_obs_t)
            advantages, returns = _compute_gae(
                rollout.rewards,
                rollout.values,
                rollout.dones,
                last_value.item(),
                cfg.gamma,
                cfg.gae_lambda,
            )
            self._update(rollout, advantages, returns)

        return self.mask_net

    def _update(self, rollout, advantages, returns):
        cfg = self.cfg
        obs_t = torch.as_tensor(rollout.obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(
            rollout.actions, dtype=torch.long, device=self.device
        )
        old_log_probs = torch.as_tensor(
            rollout.log_probs, dtype=torch.float32, device=self.device
        )
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = obs_t.shape[0]
        idx = np.arange(n)
        for _ in range(cfg.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, cfg.batch_size):
                mb = idx[start : start + cfg.batch_size]
                dist, values = self.mask_net(obs_t[mb])
                new_log_probs = dist.log_prob(actions_t[mb])
                ratio = torch.exp(new_log_probs - old_log_probs[mb])

                surr1 = ratio * adv_t[mb]
                surr2 = (
                    torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
                    * adv_t[mb]
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, ret_t[mb])
                entropy_bonus = dist.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - 0.0 * entropy_bonus

                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.mask_net.parameters(), cfg.max_grad_norm
                )
                self.optim.step()


class _Rollout:
    def __init__(self, capacity: int, obs_shape):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self.log_probs = np.zeros((capacity,), dtype=np.float32)
        self._i = 0

    def reset(self):
        self._i = 0

    def add(self, obs, action, reward, done, value, log_prob):
        i = self._i
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.values[i] = value
        self.log_probs[i] = log_prob
        self._i += 1
