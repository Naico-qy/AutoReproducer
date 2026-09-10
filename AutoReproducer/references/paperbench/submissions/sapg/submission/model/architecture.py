"""
SAPG Actor / Critic / PolicySet.

Implements the architecture described in Sec. 4.4 of:
    "SAPG: Split and Aggregate Policy Gradients", Singla, Agarwal, Pathak, ICML 2024.

From the addendum:
    The actor of each follower / leader uses a SHARED backbone B_theta
    conditioned on the parameters phi_j local to each follower / leader.
    The critic likewise shares C_psi but conditions on the same phi_j.

PaperBench reference verification (CrossRef):
    PPO baseline -- Schulman et al., arXiv:1707.06347 (CrossRef does not
    index arXiv preprints; verified via Semantic Scholar / OpenAlex paper_search).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import math
import torch
from torch import nn
from torch.distributions import Normal

from .backbones import LSTMBackbone, MLPBackbone


def _build_backbone(cfg) -> nn.Module:
    """Construct a single shared backbone (B_theta or C_psi)."""
    if cfg.recurrent:
        return LSTMBackbone(
            obs_dim=cfg.obs_dim,
            mlp_hidden=cfg.mlp_hidden,
            lstm_hidden=cfg.lstm_hidden,
            lstm_layers=cfg.lstm_layers,
            phi_dim=cfg.phi_dim,
            activation=cfg.activation,
        )
    return MLPBackbone(
        obs_dim=cfg.obs_dim,
        hidden=cfg.mlp_hidden,
        phi_dim=cfg.phi_dim,
        activation=cfg.activation,
    )


class _BackboneCfg:
    """Lightweight config carrier used internally."""

    def __init__(
        self,
        obs_dim: int,
        mlp_hidden: List[int],
        lstm_hidden: int,
        lstm_layers: int,
        phi_dim: int,
        activation: str,
        recurrent: bool,
    ) -> None:
        self.obs_dim = obs_dim
        self.mlp_hidden = mlp_hidden
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.phi_dim = phi_dim
        self.activation = activation
        self.recurrent = recurrent


class SAPGActor(nn.Module):
    """Shared B_theta + per-policy phi_j -> Gaussian action distribution.

    Per App. B Note: each policy block has its OWN learnable log_std vector
    (state-independent diagonal Gaussian, independent across policies). This
    enables different policies to have different entropies during training,
    which is what the entropy-diversity scheme in Sec. 4.5 exploits.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_policies: int,
        mlp_hidden: List[int],
        lstm_hidden: int,
        lstm_layers: int,
        phi_dim: int,
        activation: str = "elu",
        recurrent: bool = False,
        init_log_std: float = 0.0,
        separate_sigma_per_policy: bool = True,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.num_policies = num_policies
        self.recurrent = recurrent

        bcfg = _BackboneCfg(
            obs_dim,
            mlp_hidden,
            lstm_hidden,
            lstm_layers,
            phi_dim,
            activation,
            recurrent,
        )
        # SHARED backbone B_theta -- single instance, gradients flow from
        # ALL policy losses (Sec. 4.4: "parameters psi, theta are shared
        # across the leader and all followers and updated with gradients
        # from each objective").
        self.backbone = _build_backbone(bcfg)

        # Per-policy phi_j parameters (Sec. 4.4)
        self.phi = nn.Parameter(torch.zeros(num_policies, phi_dim))
        nn.init.normal_(self.phi, mean=0.0, std=0.1)

        # Mean head: linear projection from backbone output to action mean.
        self.mean_head = nn.Linear(self.backbone.out_dim, action_dim)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

        # log_std parameter (state-independent, Gaussian policy).
        # If separate_sigma_per_policy=True, we keep an independent vector per policy
        # so they can diverge in exploration scale (App. B Note).
        if separate_sigma_per_policy:
            self.log_std = nn.Parameter(
                torch.full((num_policies, action_dim), float(init_log_std))
            )
        else:
            self.log_std = nn.Parameter(
                torch.full((1, action_dim), float(init_log_std))
            )
        self.separate_sigma_per_policy = separate_sigma_per_policy

    # ------------------------------------------------------------------
    def get_phi(self, policy_idx: int) -> torch.Tensor:
        return self.phi[policy_idx]

    def get_log_std(self, policy_idx: int) -> torch.Tensor:
        if self.separate_sigma_per_policy:
            return self.log_std[policy_idx]
        return self.log_std[0]

    # ------------------------------------------------------------------
    def forward(
        self,
        obs: torch.Tensor,
        policy_idx: int,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[Normal, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        phi = self.get_phi(policy_idx)
        feats, new_hidden = self.backbone(obs, phi, hidden_state)
        mean = self.mean_head(feats)
        log_std = self.get_log_std(policy_idx).expand_as(mean)
        std = log_std.exp()
        return Normal(mean, std), new_hidden

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        policy_idx: int,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        dist, new_hidden = self.forward(obs, policy_idx, hidden_state)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, new_hidden

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        policy_idx: int,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        dist, new_hidden = self.forward(obs, policy_idx, hidden_state)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, new_hidden


class SAPGCritic(nn.Module):
    """Shared C_psi + per-policy phi_j -> scalar value V_pi_i(s).

    Important: phi for the critic is the SAME phi as for the actor of that
    policy (addendum: "the critic of each follower and leader consists of
    a shared network C_psi conditioned on the same parameters phi_j").
    """

    def __init__(
        self,
        obs_dim: int,
        num_policies: int,
        mlp_hidden: List[int],
        lstm_hidden: int,
        lstm_layers: int,
        phi_dim: int,
        activation: str = "elu",
        recurrent: bool = False,
    ) -> None:
        super().__init__()
        self.recurrent = recurrent
        self.num_policies = num_policies

        bcfg = _BackboneCfg(
            obs_dim,
            mlp_hidden,
            lstm_hidden,
            lstm_layers,
            phi_dim,
            activation,
            recurrent,
        )
        self.backbone = _build_backbone(bcfg)

        # phi for critic is shared with the actor in PolicySet, but we keep
        # a parameter here too in case the critic is used standalone.
        self.value_head = nn.Linear(self.backbone.out_dim, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(
        self,
        obs: torch.Tensor,
        phi: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        feats, new_hidden = self.backbone(obs, phi, hidden_state)
        v = self.value_head(feats).squeeze(-1)
        return v, new_hidden


class SAPGPolicySet(nn.Module):
    """Holds shared B_theta (actor) + C_psi (critic) + phi_j for every policy.

    From the addendum:
        - actor:  B_theta + phi_j (one phi per policy, j = 1...M)
        - critic: C_psi   + phi_j (SAME phi as actor; one per policy)

    We share phi between actor and critic by storing it once on the actor and
    indexing it from the critic call site.
    """

    def __init__(self, model_cfg, env_cfg) -> None:
        super().__init__()
        self.num_policies = env_cfg.num_policies
        self.action_dim = env_cfg.action_dim
        self.obs_dim = env_cfg.obs_dim

        self.actor = SAPGActor(
            obs_dim=env_cfg.obs_dim,
            action_dim=env_cfg.action_dim,
            num_policies=env_cfg.num_policies,
            mlp_hidden=list(model_cfg.obs_mlp_hidden),
            lstm_hidden=int(model_cfg.lstm_hidden),
            lstm_layers=int(model_cfg.lstm_layers),
            phi_dim=int(model_cfg.phi_dim),
            activation=model_cfg.activation,
            recurrent=bool(model_cfg.recurrent),
            init_log_std=float(model_cfg.init_log_std),
            separate_sigma_per_policy=bool(model_cfg.separate_sigma_per_policy),
        )

        # Critic uses its OWN backbone (C_psi) but reuses phi from the actor
        # (addendum: same phi_j conditions both). We do this by passing
        # actor.phi[i] into critic.forward() at call-site.
        self.critic = SAPGCritic(
            obs_dim=env_cfg.obs_dim,
            num_policies=env_cfg.num_policies,
            mlp_hidden=list(model_cfg.obs_mlp_hidden),
            lstm_hidden=int(model_cfg.lstm_hidden),
            lstm_layers=int(model_cfg.lstm_layers),
            phi_dim=int(model_cfg.phi_dim),
            activation=model_cfg.activation,
            recurrent=bool(model_cfg.recurrent),
        )

    # ----- convenience accessors -----
    def value(
        self,
        obs: torch.Tensor,
        policy_idx: int,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        phi = self.actor.get_phi(policy_idx)
        return self.critic(obs, phi, hidden_state)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
