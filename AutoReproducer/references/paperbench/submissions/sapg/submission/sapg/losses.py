"""
SAPG losses -- direct implementations of the equations in the paper.

Equation references match Singla, Agarwal, Pathak (ICML 2024).

  Eq. 2  L_on(pi_theta)        -- PPO clipped surrogate (Schulman et al., 2017)
  Eq. 3  L_off(pi_i; X)        -- IS-corrected off-policy PPO surrogate
  Eq. 4  L = L_on + lambda * L_off
  Eq. 5  V_target(s_t)          n-step return for on-policy critic
  Eq. 6  V_target(s'_t)         1-step return for off-policy critic
  Eq. 7  L_critic_on
  Eq. 8  L_critic_off
  Eq. 9  L_critic = L_on + lambda * L_off

In the leader-follower variant (Sec. 4.3) the leader's set X = {2..M} and
followers' X = {} (so they reduce to vanilla PPO + entropy).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


# =============================================================================
# Building blocks
# =============================================================================


def importance_ratio(new_logp: torch.Tensor, old_logp: torch.Tensor) -> torch.Tensor:
    """r_t(pi_theta) = pi_theta(a|s) / pi_old(a|s) computed in log-space."""
    return torch.exp(new_logp - old_logp)


def _ppo_clip(
    adv: torch.Tensor, ratio: torch.Tensor, lo: float, hi: float
) -> torch.Tensor:
    """min(r * A, clip(r, lo, hi) * A)  -- maximised, so loss = -mean(...)."""
    return torch.min(ratio * adv, ratio.clamp(lo, hi) * adv)


# =============================================================================
# Eq. 2 -- on-policy PPO loss
# =============================================================================


def on_policy_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
) -> torch.Tensor:
    ratio = importance_ratio(new_logp, old_logp)
    surrogate = _ppo_clip(advantages, ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    return -surrogate.mean()


# =============================================================================
# Eq. 3 -- off-policy PPO loss with IS correction
# =============================================================================


def off_policy_loss(
    new_logp: torch.Tensor,  # log pi_i(a|s)
    behavior_logp: torch.Tensor,  # log pi_j(a|s)  -- the data-collecting policy
    old_logp_i: torch.Tensor,  # log pi_i,old(a|s)
    advantages: torch.Tensor,  # A^{pi_i,old}(s, a)
    clip_eps: float,
) -> torch.Tensor:
    """
    L_off(pi_i; X) = (1/|X|) sum_{j in X} E_{(s,a)~pi_j} [
        min( r_pi_i(s,a),
             clip( r_pi_i(s,a), mu*(1-eps), mu*(1+eps) ) ) * A^{pi_i,old}
    ]

    where:
        r_pi_i(s,a) = pi_i(a|s) / pi_j(a|s)
        mu          = pi_i,old(a|s) / pi_j(a|s)
    """
    # r_pi_i = pi_i / pi_j
    r = torch.exp(new_logp - behavior_logp)
    # mu = pi_i,old / pi_j
    mu = torch.exp(old_logp_i - behavior_logp)

    lo = mu * (1.0 - clip_eps)
    hi = mu * (1.0 + clip_eps)
    surrogate = torch.min(r * advantages, torch.clamp(r, lo, hi) * advantages)
    return -surrogate.mean()


# =============================================================================
# Eqs. 7-9 -- critic losses
# =============================================================================


def on_policy_critic_loss(
    values: torch.Tensor, n_step_targets: torch.Tensor
) -> torch.Tensor:
    """L_on^critic = E[(V_pi_i(s) - V_target_n)^2]   (Eq. 7)."""
    return ((values - n_step_targets) ** 2).mean()


def off_policy_critic_loss(
    values: torch.Tensor,
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """L_off^critic uses the 1-step bootstrap target (Eq. 6 / Eq. 8).

    V_target_off(s'_t) = r_t + gamma * V_pi_j,old(s_{t+1}')
    """
    target = rewards + gamma * (1.0 - dones) * next_values
    return ((values - target.detach()) ** 2).mean()


# =============================================================================
# Eq. 4 / Eq. 9 -- combined losses (leader-follower variant)
# =============================================================================


def sapg_combined_actor_loss(
    on_terms: Dict[str, torch.Tensor],
    off_terms: Optional[Dict[str, torch.Tensor]],
    clip_eps: float,
    off_lambda: float,
    is_leader: bool,
    entropy: torch.Tensor,
    base_entropy_coef: float,
    diversity_entropy_coef: float,
    bounds_coef: float,
    action_mean: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Returns a dict containing the actor loss and its components.

    Followers (is_leader=False, off_terms=None):
        L = L_on + (base + diversity)*(-H)  + bounds_coef*bounds  -- Sec 4.5

    Leader (is_leader=True, off_terms supplied):
        L = L_on + lambda * L_off + base*(-H) + bounds_coef*bounds  -- Eq. 4
    """
    on_loss = on_policy_loss(
        on_terms["new_logp"], on_terms["old_logp"], on_terms["advantages"], clip_eps
    )

    off_loss = torch.zeros((), device=on_loss.device)
    if is_leader and off_terms is not None and off_terms["new_logp"].numel() > 0:
        off_loss = off_policy_loss(
            new_logp=off_terms["new_logp"],
            behavior_logp=off_terms["behavior_logp"],
            old_logp_i=off_terms["old_logp_i"],
            advantages=off_terms["advantages"],
            clip_eps=clip_eps,
        )

    # entropy bonus (NOTE: paper writes "L = L_on + sigma*(j-1)*H" so we
    # subtract the entropy from the loss to maximise it).
    eff_entropy_coef = base_entropy_coef
    if not is_leader:
        eff_entropy_coef = base_entropy_coef + diversity_entropy_coef
    ent_loss = -eff_entropy_coef * entropy.mean()

    # bounds loss penalises action means leaving [-1, 1]
    b_loss = torch.zeros((), device=on_loss.device)
    if action_mean is not None and bounds_coef > 0:
        b_loss = bounds_loss(action_mean) * bounds_coef

    total = on_loss + off_lambda * off_loss + ent_loss + b_loss

    return {
        "actor_loss": total,
        "on_loss": on_loss.detach(),
        "off_loss": off_loss.detach(),
        "ent_loss": ent_loss.detach(),
        "bounds_loss": b_loss.detach(),
        "entropy": entropy.mean().detach(),
    }


def sapg_combined_critic_loss(
    on_values: torch.Tensor,
    on_targets: torch.Tensor,
    off_values: Optional[torch.Tensor],
    off_rewards: Optional[torch.Tensor],
    off_next_values: Optional[torch.Tensor],
    off_dones: Optional[torch.Tensor],
    off_lambda: float,
    gamma: float,
    is_leader: bool,
    critic_coef: float,
) -> Dict[str, torch.Tensor]:
    """L_critic = critic_coef * (L_on^critic + lambda * L_off^critic)."""
    l_on = on_policy_critic_loss(on_values, on_targets)
    l_off = torch.zeros((), device=on_values.device)
    if is_leader and off_values is not None and off_values.numel() > 0:
        l_off = off_policy_critic_loss(
            off_values, off_rewards, off_next_values, off_dones, gamma
        )
    total = critic_coef * (l_on + off_lambda * l_off)
    return {
        "critic_loss": total,
        "critic_on": l_on.detach(),
        "critic_off": l_off.detach(),
    }


# =============================================================================
# Bounds loss (App. B Table 2: bounds loss coefficient = 1e-4)
# =============================================================================


def bounds_loss(action_mean: torch.Tensor, bound: float = 1.1) -> torch.Tensor:
    """Penalty pushing action means inside [-bound, bound].

    Used by IsaacGymEnvs PPO implementations and Petrenko et al. (2023).
    """
    high = (action_mean - bound).clamp(min=0.0)
    low = (-bound - action_mean).clamp(min=0.0)
    return (high.pow(2) + low.pow(2)).mean()
