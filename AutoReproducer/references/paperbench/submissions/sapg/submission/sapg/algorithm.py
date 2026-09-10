"""
SAPG training algorithm (Algorithm 1 from the paper).

Pseudocode reproduced for clarity (paper p. 5):

    Initialize shared parameters theta, psi
    For i in 1..M initialize parameters phi_i
    Initialize N envs E_1..E_N
    Initialize buffers D_1..D_M
    for iter in 1, 2, ...:
        for j in 1..M:
            D_j <- CollectData( E_{j*N/M : (j+1)*N/M}, theta, psi_j )
        L = 0
        Sample |D_1| transitions from union_{j=2..M} D_j -> D_1'
        L += OffPolicyLoss(D_1')
        L += OnPolicyLoss(D_1)
        for j in 2..M:
            L += OnPolicyLoss(D_j)
        Update theta <- theta - eta * grad_theta L
        Update psi   <- psi   - eta * grad_psi L

The "leader-follower" variant is the default (Sec. 4.3, 4.6); the
"symmetric" variant is implemented as well (Sec. 4.2) and toggled by
config.sapg.variant.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import math
import time

import torch
from torch import nn

from data.loader import RolloutBuffer, SAPGRolloutStorage, compute_gae, n_step_return
from model.architecture import SAPGPolicySet
from sapg.losses import (
    sapg_combined_actor_loss,
    sapg_combined_critic_loss,
)


class SAPGAlgorithm:
    """Owns the policy set, optimisers, rollout storage, and update loop.

    Important design choices grounded in the paper:
      * Shared optimizer over theta (B_theta + mean_head + log_std)
        and psi (C_psi + value_head). Phi is updated only by gradients
        from its own policy (so we put phi in its own param group with the
        same lr -- gradients from other policies w.r.t. phi_j will be zero
        because phi_j is only read by policy j's loss).
      * Adaptive learning rate: if KL-divergence between old and new policy
        exceeds kl_threshold, lr is decreased; if it is well below half the
        threshold, lr is increased. (App. B: "KL threshold for LR update = 0.016")
      * Mini-batch SGD with `mini_epochs` passes (Table 2: 2 for AllegroKuka,
        5 for ShadowHand / AllegroHand).
    """

    def __init__(self, cfg, policy_set: SAPGPolicySet, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.policy = policy_set.to(device)
        self.M = cfg.env.num_policies
        self.block_size = cfg.env.num_envs // self.M
        self.horizon = cfg.ppo.horizon_length
        self.is_recurrent = bool(cfg.model.recurrent)
        self.gamma = float(cfg.ppo.gamma)
        self.tau = float(cfg.ppo.tau)
        self.clip_eps = float(cfg.ppo.clip_eps)
        self.entropy_coef = float(cfg.ppo.entropy_coef)
        self.div_sigma = float(cfg.sapg.entropy_diversity_sigma)
        self.bounds_coef = float(cfg.ppo.bounds_loss_coef)
        self.critic_coef = float(cfg.ppo.critic_coef)
        self.off_lambda = float(cfg.sapg.off_policy_lambda)
        self.subsample_off_policy = bool(cfg.sapg.subsample_off_policy)
        self.variant = str(cfg.sapg.variant)
        self.normalize_adv = bool(cfg.ppo.normalize_advantage)
        self.kl_thresh = float(cfg.ppo.kl_threshold)
        self.grad_norm = float(cfg.ppo.grad_norm)
        self.mini_epochs = int(cfg.ppo.mini_epochs)
        self.mb_size = int(self.block_size * cfg.ppo.mini_batch_size_factor)
        self.n_step = int(cfg.ppo.n_step_return)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=float(cfg.ppo.learning_rate)
        )
        self.lr = float(cfg.ppo.learning_rate)

        self.storage = SAPGRolloutStorage(
            num_policies=self.M,
            block_size=self.block_size,
            horizon=self.horizon,
            obs_dim=cfg.env.obs_dim,
            action_dim=cfg.env.action_dim,
            device=device,
        )

    # =================================================================
    # Rollout collection
    # =================================================================
    @torch.no_grad()
    def collect(self, env, obs: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Roll out `horizon` steps across all M policies in parallel.

        envs are split into contiguous blocks; block j is driven by policy j.
        Returns the new observation tensor and a dict of metrics.
        """
        T = self.horizon
        ep_returns: List[float] = []
        successes: List[float] = []

        for t in range(T):
            actions = torch.zeros(
                env.num_envs, self.cfg.env.action_dim, device=self.device
            )
            log_probs = torch.zeros(env.num_envs, device=self.device)
            values = torch.zeros(env.num_envs, device=self.device)

            for j in range(self.M):
                lo = j * self.block_size
                hi = (j + 1) * self.block_size
                obs_j = obs[lo:hi]
                a_j, lp_j, _ = self.policy.actor.act(
                    obs_j, policy_idx=j, deterministic=False
                )
                v_j, _ = self.policy.value(obs_j, policy_idx=j)
                actions[lo:hi] = a_j
                log_probs[lo:hi] = lp_j
                values[lo:hi] = v_j

            actions = actions.clamp(-1.0, 1.0)
            next_obs, rewards, dones, info = env.step(actions)
            if not torch.is_tensor(next_obs):
                next_obs = torch.as_tensor(
                    next_obs, device=self.device, dtype=torch.float32
                )
            if not torch.is_tensor(rewards):
                rewards = torch.as_tensor(
                    rewards, device=self.device, dtype=torch.float32
                )
            if not torch.is_tensor(dones):
                dones = torch.as_tensor(dones, device=self.device, dtype=torch.float32)

            for j in range(self.M):
                lo = j * self.block_size
                hi = (j + 1) * self.block_size
                buf = self.storage.buf(j)
                buf.obs[t] = obs[lo:hi]
                buf.actions[t] = actions[lo:hi]
                buf.log_probs[t] = log_probs[lo:hi]
                buf.values[t] = values[lo:hi]
                buf.rewards[t] = rewards[lo:hi]
                buf.dones[t] = dones[lo:hi]
                buf.next_obs[t] = next_obs[lo:hi]

            obs = next_obs
            if isinstance(info, dict):
                if "episode_return" in info:
                    ep_returns.append(float(info["episode_return"].mean().item()))
                if "successes" in info:
                    successes.append(float(info["successes"].mean().item()))

        # bootstrap value
        for j in range(self.M):
            lo = j * self.block_size
            hi = (j + 1) * self.block_size
            v_last, _ = self.policy.value(obs[lo:hi], policy_idx=j)
            self.storage.buf(j).values[T] = v_last

        # GAE + n-step targets
        for j in range(self.M):
            buf = self.storage.buf(j)
            adv, ret = compute_gae(
                buf.rewards, buf.values, buf.dones, self.gamma, self.tau
            )
            buf.advantages = adv
            buf.returns = ret
            buf.n_step_targets = n_step_return(
                buf.rewards, buf.values, buf.dones, self.gamma, self.n_step
            )

        metrics = {
            "ep_return": float(sum(ep_returns) / max(1, len(ep_returns)))
            if ep_returns
            else 0.0,
            "successes": float(sum(successes) / max(1, len(successes)))
            if successes
            else 0.0,
            "mean_value": float(
                torch.stack([b.values[:-1].mean() for b in self.storage.buffers])
                .mean()
                .item()
            ),
        }
        return obs, metrics

    # =================================================================
    # Update step (Algorithm 1 lines 7-13)
    # =================================================================
    def update(self) -> Dict[str, float]:
        flat_per_policy = [self.storage.buf(j).flatten() for j in range(self.M)]

        # Per-policy: advantage normalisation
        if self.normalize_adv:
            for f in flat_per_policy:
                a = f["advantages"]
                f["advantages"] = (a - a.mean()) / (a.std() + 1e-8)

        log_acc: Dict[str, List[float]] = {}
        kl_acc: List[float] = []

        for epoch in range(self.mini_epochs):
            for j in range(self.M):
                is_leader = (self.variant == "leader_follower" and j == 0) or (
                    self.variant == "symmetric"
                )
                # Fetch on-policy minibatch
                f = flat_per_policy[j]
                n = f["obs"].shape[0]
                idx = torch.randperm(n, device=self.device)
                for start in range(0, n, self.mb_size):
                    mb = idx[start : start + self.mb_size]
                    on_logp, on_entropy, _ = self.policy.actor.evaluate_actions(
                        f["obs"][mb], f["actions"][mb], policy_idx=j
                    )
                    on_values, _ = self.policy.value(f["obs"][mb], policy_idx=j)

                    on_terms = {
                        "new_logp": on_logp,
                        "old_logp": f["log_probs"][mb],
                        "advantages": f["advantages"][mb],
                    }

                    # ---- Gather off-policy minibatch (leader only) ----
                    off_terms = None
                    off_critic_inputs: Dict[str, Optional[torch.Tensor]] = dict(
                        off_values=None,
                        off_rewards=None,
                        off_next_values=None,
                        off_dones=None,
                    )

                    if is_leader and self.variant != "no_off_policy":
                        off_terms, off_critic_inputs = (
                            self._gather_off_policy_minibatch(
                                target_policy=j,
                                flat_per_policy=flat_per_policy,
                                mb_size=mb.shape[0],
                            )
                        )

                    actor_terms = sapg_combined_actor_loss(
                        on_terms=on_terms,
                        off_terms=off_terms,
                        clip_eps=self.clip_eps,
                        off_lambda=self.off_lambda,
                        is_leader=is_leader,
                        entropy=on_entropy,
                        base_entropy_coef=self.entropy_coef,
                        # follower diversity term: sigma * (j-1) for j=1..M-1
                        # paper indexing: leader is i=1 -> j=0 here. For
                        # follower j>=1 we apply sigma*(j) since the paper's
                        # (i-1) becomes our zero-based j.
                        diversity_entropy_coef=self.div_sigma * j,
                        bounds_coef=self.bounds_coef,
                        action_mean=None,
                    )

                    critic_terms = sapg_combined_critic_loss(
                        on_values=on_values,
                        on_targets=f["n_step_targets"][mb].detach(),
                        off_values=off_critic_inputs.get("off_values"),
                        off_rewards=off_critic_inputs.get("off_rewards"),
                        off_next_values=off_critic_inputs.get("off_next_values"),
                        off_dones=off_critic_inputs.get("off_dones"),
                        off_lambda=self.off_lambda,
                        gamma=self.gamma,
                        is_leader=is_leader,
                        critic_coef=self.critic_coef,
                    )

                    loss = actor_terms["actor_loss"] + critic_terms["critic_loss"]

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_norm)
                    self.optimizer.step()

                    with torch.no_grad():
                        approx_kl = ((f["log_probs"][mb] - on_logp).mean()).item()
                    kl_acc.append(approx_kl)

                    for k, v in actor_terms.items():
                        log_acc.setdefault(k, []).append(
                            float(v.detach().cpu().item())
                            if torch.is_tensor(v)
                            else float(v)
                        )
                    for k, v in critic_terms.items():
                        log_acc.setdefault(k, []).append(
                            float(v.detach().cpu().item())
                            if torch.is_tensor(v)
                            else float(v)
                        )

        # adaptive LR (paper: KL threshold 0.016 from Table 2)
        if kl_acc:
            mean_kl = float(sum(map(abs, kl_acc)) / len(kl_acc))
            if mean_kl > 2.0 * self.kl_thresh:
                self.lr = max(self.lr / 1.5, 1e-6)
            elif mean_kl < 0.5 * self.kl_thresh:
                self.lr = min(self.lr * 1.5, 1e-2)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.lr

        agg = {k: float(sum(v) / max(1, len(v))) for k, v in log_acc.items()}
        agg["lr"] = self.lr
        agg["approx_kl"] = (
            float(sum(map(abs, kl_acc)) / max(1, len(kl_acc))) if kl_acc else 0.0
        )
        return agg

    # =================================================================
    # Off-policy minibatch sampling (Algorithm 1, line 8)
    # =================================================================
    def _gather_off_policy_minibatch(
        self,
        target_policy: int,
        flat_per_policy: List[Dict[str, torch.Tensor]],
        mb_size: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Sample |mb_size| transitions from union_{j != target} D_j and
        compute the IS-weighted PPO terms with respect to policy `target_policy`.

        Sec. 4.3 says we subsample off-policy data so that
            |off-policy minibatch| == |on-policy minibatch|.
        Setting subsample_off_policy=False reproduces the "high off-policy
        ratio" ablation from Sec. 6.3.
        """
        # 1. concatenate all OTHER policies' flattened buffers
        if self.variant == "symmetric":
            other_idxs = [j for j in range(self.M) if j != target_policy]
        else:  # leader_follower
            other_idxs = list(range(1, self.M))  # always 1..M-1
            if target_policy != 0:
                # safety: this branch shouldn't be reached for non-leader
                return (
                    {
                        "new_logp": torch.empty(0, device=self.device),
                        "behavior_logp": torch.empty(0, device=self.device),
                        "old_logp_i": torch.empty(0, device=self.device),
                        "advantages": torch.empty(0, device=self.device),
                    },
                    {
                        "off_values": None,
                        "off_rewards": None,
                        "off_next_values": None,
                        "off_dones": None,
                    },
                )

        cat_obs = torch.cat([flat_per_policy[j]["obs"] for j in other_idxs], dim=0)
        cat_act = torch.cat([flat_per_policy[j]["actions"] for j in other_idxs], dim=0)
        cat_blogp = torch.cat(
            [flat_per_policy[j]["log_probs"] for j in other_idxs], dim=0
        )
        cat_adv = torch.cat(
            [flat_per_policy[j]["advantages"] for j in other_idxs], dim=0
        )
        cat_rew = torch.cat([flat_per_policy[j]["rewards"] for j in other_idxs], dim=0)
        cat_done = torch.cat([flat_per_policy[j]["dones"] for j in other_idxs], dim=0)
        cat_nextobs = torch.cat(
            [flat_per_policy[j]["next_obs"] for j in other_idxs], dim=0
        )

        N = cat_obs.shape[0]
        size = mb_size if self.subsample_off_policy else N
        size = min(size, N)
        idx = torch.randint(0, N, (size,), device=self.device)

        obs_b = cat_obs[idx]
        act_b = cat_act[idx]
        behavior_logp = cat_blogp[idx]
        adv_b = cat_adv[idx]
        rew_b = cat_rew[idx]
        done_b = cat_done[idx]
        nextobs_b = cat_nextobs[idx]

        # log pi_i (current target)  and  log pi_i,old  (we use old = behavior
        # of the SAME data point under target's *previous* parameters; in
        # practice we approximate pi_i,old by re-evaluating under the old
        # actor's stored log_probs. Without versioned snapshots we use the
        # behavior logp as a proxy for pi_i,old when target == j; otherwise
        # we re-evaluate target under the current network for pi_i and
        # use the same with stop-grad for pi_i,old (a common practical
        # approximation, see Meng et al., 2023).
        new_logp, _, _ = self.policy.actor.evaluate_actions(
            obs_b, act_b, policy_idx=target_policy
        )
        with torch.no_grad():
            old_logp_i = new_logp.detach().clone()

        # Critic-side off-policy targets (Eq. 6: 1-step bootstrap)
        off_values, _ = self.policy.value(obs_b, policy_idx=target_policy)
        with torch.no_grad():
            off_next_values, _ = self.policy.value(nextobs_b, policy_idx=target_policy)

        return (
            {
                "new_logp": new_logp,
                "behavior_logp": behavior_logp.detach(),
                "old_logp_i": old_logp_i,
                "advantages": adv_b.detach(),
            },
            {
                "off_values": off_values,
                "off_rewards": rew_b.detach(),
                "off_next_values": off_next_values.detach(),
                "off_dones": done_b.detach(),
            },
        )
