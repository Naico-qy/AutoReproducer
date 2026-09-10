"""Actor-Critic and Mask network architectures.

Per the addendum, the authors used Stable-Baselines3's default ``MlpPolicy``
for the dense/sparse MuJoCo experiments. We replicate that default below for
the cases where SB3 cannot be used directly (e.g. when we need to expose
intermediate gradients to the RND head). The classes here are intentionally
simple and SB3-compatible: ``ActorCritic`` produces ``(distribution, value)``
pairs in the same convention as ``stable_baselines3.common.policies``.

Variable names mirror the paper's notation:
    π(a|s)   -> ActorCritic.actor
    V^π(s)   -> ActorCritic.critic
    π̃(a^m|s) -> MaskNet.actor   (binary policy: 0=keep, 1=blind)
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def _mlp(
    in_dim: int, out_dim: int, hidden: Sequence[int], activation: type = nn.Tanh
) -> nn.Sequential:
    layers = []
    last = in_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        layers.append(activation())
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Generic actor-critic MLP matching SB3 ``MlpPolicy`` defaults.

    For continuous action spaces (MuJoCo) the actor outputs a Gaussian with
    state-independent log-std (SB3 default). For discrete spaces it outputs
    Categorical logits.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: Sequence[int] = (64, 64),
        continuous: bool = True,
        init_log_std: float = 0.0,
    ):
        super().__init__()
        self.continuous = continuous
        self.actor = _mlp(obs_dim, action_dim, hidden)
        self.critic = _mlp(obs_dim, 1, hidden)
        if continuous:
            self.log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)

    def forward(self, obs: torch.Tensor):
        mean_or_logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        if self.continuous:
            std = self.log_std.exp().expand_as(mean_or_logits)
            dist = Normal(mean_or_logits, std)
        else:
            dist = Categorical(logits=mean_or_logits)
        return dist, value

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist, value = self.forward(obs)
        if deterministic:
            if self.continuous:
                action = dist.mean
            else:
                action = dist.probs.argmax(-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        if self.continuous:
            log_prob = log_prob.sum(-1)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        dist, value = self.forward(obs)
        log_prob = dist.log_prob(action)
        if self.continuous:
            log_prob = log_prob.sum(-1)
        entropy = dist.entropy()
        if self.continuous:
            entropy = entropy.sum(-1)
        return log_prob, value, entropy


class MaskNet(nn.Module):
    """Binary mask network (paper §3.3 Eq. 1).

    The mask net consumes a state ``s_t`` and outputs ``a^m_t ∈ {0, 1}``.
    A value of 1 means "blind the target agent" (substitute its action with a
    random one); 0 means "keep the target agent's action".
    """

    def __init__(self, obs_dim: int, hidden: Sequence[int] = (64, 64)):
        super().__init__()
        self.actor = _mlp(obs_dim, 2, hidden)  # binary categorical
        self.critic = _mlp(obs_dim, 1, hidden)

    def forward(self, obs: torch.Tensor):
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return Categorical(logits=logits), value

    def importance(self, obs: torch.Tensor) -> torch.Tensor:
        """Per the paper: state importance = P(mask outputs 0).

        Higher value -> more important, since blinding (output=1) at this
        step would not change the final reward, hence the step matters.
        Wait — actually the paper defines importance as the *probability the
        mask network outputs 0* (i.e. probability that the agent's action
        is preserved): a state at which the agent must act precisely is one
        where any blinding hurts the episode reward, so the mask net learns
        to output 0 there. We follow the paper's definition exactly.
        """
        with torch.no_grad():
            logits = self.actor(obs)
            probs = torch.softmax(logits, dim=-1)
            return probs[..., 0]  # P(a^m = 0)
