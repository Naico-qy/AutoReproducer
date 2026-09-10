"""RICE refinement loop (Algorithm 2 of the paper).

Implements the full pseudocode from §3.3:

    for iteration = 1, 2, ...
        D <- ∅
        RAND_NUM ~ U(0,1)
        if RAND_NUM < p:
            run π to obtain trajectory τ of length K
            identify the most critical state s_t in τ via state mask π̃
            set initial state s0 <- s_t                   ← d^π̂_ρ(s) sample
        else:
            set initial state s0 ~ ρ                       ← default
        for t = 0 to T:
            sample a_t ~ π(a|s_t)
            (s_{t+1}, R_t) <- env.step(a_t)
            R^RND_t = ||f(s_{t+1}) - f̂(s_{t+1})||²        ← normalised
            add (s_t, s_{t+1}, a_t, R_t + λ·R^RND_t) to D
        optimise π_θ on D via PPO
        optimise f̂_θ on D via MSE (Adam)

Therefore this module orchestrates: (1) the explanation-based reset, (2) the
PPO update on the augmented reward, and (3) the RND predictor update — using
the components implemented in ``model/architecture.py``, ``model/rnd.py`` and
``model/explanation.py``.

The hyper-parameters ``p, β, λ, α`` align with Experiment V of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .architecture import ActorCritic, MaskNet
from .rnd import RNDModule
from .explanation import identify_critical_state


@dataclass
class RefineConfig:
    total_timesteps: int = 1_000_000
    reset_probability: float = 0.5  # p
    beta: float = 0.5  # μ(s) mixing weight (informational)
    rnd_lambda: float = 0.01  # λ
    rnd_lr: float = 1e-4
    rnd_feature_dim: int = 128
    rnd_hidden: tuple = (128, 128)
    rnd_normalize_obs: bool = True
    rnd_normalize_reward: bool = True
    trajectory_length_K: int = 1000  # K in Algorithm 2
    learning_rate: float = 3e-5
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    max_grad_norm: float = 0.5
    device: str = "cpu"
    explanation_K: float = 0.1  # K for fidelity / critical-state window


def _gae(rewards, values, dones, last_value, gamma, gae_lambda):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last = 0.0
    n = len(rewards)
    for t in reversed(range(n)):
        next_value = last_value if t == n - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last = delta + gamma * gae_lambda * non_terminal * last
        advantages[t] = last
    returns = advantages + values
    return advantages, returns


class RICERefiner:
    """Algorithm 2: refine ``policy`` using mask-net explanations + RND."""

    def __init__(self, env, policy: ActorCritic, mask_net: MaskNet, cfg: RefineConfig):
        self.env = env
        self.policy = policy.to(cfg.device)
        self.mask_net = mask_net.to(cfg.device).eval() if mask_net is not None else None
        self.cfg = cfg
        self.device = cfg.device

        obs_dim = int(np.prod(env.observation_space.shape))
        self.rnd = RNDModule(
            obs_dim=obs_dim,
            feature_dim=cfg.rnd_feature_dim,
            hidden=cfg.rnd_hidden,
            lr=cfg.rnd_lr,
            normalize_obs=cfg.rnd_normalize_obs,
            normalize_reward=cfg.rnd_normalize_reward,
            device=cfg.device,
        )
        self.optim = Adam(self.policy.parameters(), lr=cfg.learning_rate)

        self.action_space = env.action_space
        self.continuous_actions = hasattr(self.action_space, "high")

    # ----------------------------------------------------------- reset logic
    def _sample_initial_state(self):
        """Implements μ(s) = β·d^π̂_ρ(s) + (1-β)·ρ(s) by Bernoulli(p).

        With probability ``p`` we identify a critical state via the mask
        network and reset the env to that state; otherwise we draw ``s_0``
        from the default initial distribution ``ρ`` (a fresh ``env.reset``).
        """
        cfg = self.cfg
        rand = float(np.random.rand())
        if rand < cfg.reset_probability and self.mask_net is not None:
            critical = identify_critical_state(
                self.env,
                self.policy,
                self.mask_net,
                K=cfg.explanation_K,
                max_steps=cfg.trajectory_length_K,
                random=False,
                device=cfg.device,
            )
            obs, _ = self.env.reset()
            if (
                critical.snapshot is not None
                and hasattr(self.env, "restore")
                and self.env.restore(critical.snapshot)
            ):
                obs = critical.state
        else:
            obs, _ = self.env.reset()
        return obs

    # -------------------------------------------------------------- training
    def train(self):
        cfg = self.cfg
        rollout = _Buffer(
            cfg.n_steps, self.env.observation_space.shape, self.action_space
        )
        timestep = 0
        obs = self._sample_initial_state()

        while timestep < cfg.total_timesteps:
            rollout.reset()
            for step in range(cfg.n_steps):
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                with torch.no_grad():
                    action, log_prob, value = self.policy.act(obs_t)
                action_np = action.squeeze(0).cpu().numpy()

                next_obs, reward, term, trunc, _ = self.env.step(action_np)
                done = bool(term or trunc)

                # RND intrinsic reward (paper §3.3)
                intrinsic = float(
                    self.rnd.intrinsic_reward(np.asarray(next_obs, dtype=np.float32))
                )
                augmented = float(reward) + cfg.rnd_lambda * intrinsic

                rollout.add(
                    obs,
                    action_np,
                    augmented,
                    done,
                    value.item(),
                    log_prob.item(),
                    next_obs,
                )
                obs = next_obs
                timestep += 1
                if done:
                    obs = self._sample_initial_state()

            with torch.no_grad():
                last_obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                _, last_value = self.policy.forward(last_obs_t)
            advantages, returns = _gae(
                rollout.rewards,
                rollout.values,
                rollout.dones,
                last_value.item(),
                cfg.gamma,
                cfg.gae_lambda,
            )
            self._ppo_update(rollout, advantages, returns)
            self.rnd.update(rollout.next_obs)
        return self.policy

    # --------------------------------------------------------------- updates
    def _ppo_update(self, rollout, advantages, returns):
        cfg = self.cfg
        obs_t = torch.as_tensor(rollout.obs, dtype=torch.float32, device=self.device)
        if self.continuous_actions:
            actions_t = torch.as_tensor(
                rollout.actions, dtype=torch.float32, device=self.device
            )
        else:
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
                new_log_probs, values, entropy = self.policy.evaluate(
                    obs_t[mb], actions_t[mb]
                )
                ratio = torch.exp(new_log_probs - old_log_probs[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = (
                    torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
                    * adv_t[mb]
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, ret_t[mb])
                entropy_bonus = entropy.mean()
                loss = policy_loss + 0.5 * value_loss - 0.0 * entropy_bonus

                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), cfg.max_grad_norm
                )
                self.optim.step()


class _Buffer:
    def __init__(self, capacity, obs_shape, action_space):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        if hasattr(action_space, "shape") and len(action_space.shape) > 0:
            self.actions = np.zeros((capacity, *action_space.shape), dtype=np.float32)
        else:
            self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self.log_probs = np.zeros((capacity,), dtype=np.float32)
        self._i = 0

    def reset(self):
        self._i = 0

    def add(self, obs, action, reward, done, value, log_prob, next_obs):
        i = self._i
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.values[i] = value
        self.log_probs[i] = log_prob
        self._i += 1
