"""Step-level explanation utilities (paper §3.3 + addendum).

Two public callables:

* ``identify_critical_state(env, target_policy, mask_net, K)`` — runs a
  trajectory under ``target_policy``, scores every step's importance via the
  mask network's ``P(a^m = 0)`` head, picks the highest-scoring sliding
  window of length ``L·K``, and returns:
      (state, snapshot, importance_scores, window_start, window_length)

* ``fidelity_score(env, target_policy, mask_net, K, n_trajectories, d_max)``
  — implements the addendum's pipeline exactly:
      1. score each step,
      2. find best window of length L·K,
      3. fast-forward the agent to the start, randomise actions for the
         window's duration, then resume the policy until episode end,
      4. compute  log(d/d_max) − log(l/L)  averaged over n_trajectories.

The "Random" baseline-explanation is also exposed via
``identify_critical_state(..., random=True)`` which selects a uniformly
random visited state — used by Experiment III.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from .architecture import ActorCritic, MaskNet


@dataclass
class CriticalState:
    state: np.ndarray
    snapshot: object
    importance: np.ndarray  # full per-step importance trace
    window_start: int
    window_length: int
    trajectory_length: int


@torch.no_grad()
def _rollout_with_scores(
    env,
    policy: ActorCritic,
    mask_net: Optional[MaskNet],
    max_steps: int = 1000,
    device: str = "cpu",
):
    """Roll out ``policy`` and record per-step importance and snapshots."""
    states: List[np.ndarray] = []
    snapshots: List[object] = []
    importance: List[float] = []
    actions, rewards = [], []

    obs, _ = env.reset()
    for t in range(max_steps):
        states.append(np.asarray(obs, dtype=np.float32))
        snapshots.append(env.snapshot() if hasattr(env, "snapshot") else None)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action, _, _ = policy.act(obs_t, deterministic=False)
        action_np = action.squeeze(0).cpu().numpy()

        if mask_net is not None:
            score = mask_net.importance(obs_t).item()
        else:
            score = 0.0
        importance.append(score)

        obs, reward, term, trunc, _ = env.step(action_np)
        actions.append(action_np)
        rewards.append(float(reward))
        if term or trunc:
            break

    return (
        np.array(states),
        snapshots,
        np.array(importance, dtype=np.float32),
        np.array(actions),
        np.array(rewards, dtype=np.float32),
    )


def _best_window(scores: np.ndarray, window_len: int) -> int:
    """Return start index of the highest-mean sliding window."""
    if window_len <= 0 or window_len > len(scores):
        return 0
    cum = np.cumsum(np.insert(scores, 0, 0.0))
    means = (cum[window_len:] - cum[:-window_len]) / float(window_len)
    return int(np.argmax(means))


def identify_critical_state(
    env,
    target_policy: ActorCritic,
    mask_net: Optional[MaskNet],
    K: float = 0.1,
    max_steps: int = 1000,
    random: bool = False,
    device: str = "cpu",
) -> CriticalState:
    """Identify the most-critical state in a fresh trajectory.

    K is the sliding-window fraction (paper varies K ∈ {0.1,0.2,0.3,0.4}).
    If ``random=True`` we ignore ``mask_net`` and return a uniformly
    sampled visited state — this implements the "Random" baseline.
    """
    states, snapshots, scores, _, _ = _rollout_with_scores(
        env, target_policy, mask_net, max_steps=max_steps, device=device
    )
    L = len(states)
    window_len = max(1, int(L * K))

    if random or mask_net is None:
        # Random-explanation baseline: pick a random visited state.
        start = int(np.random.randint(0, max(1, L - window_len + 1)))
    else:
        start = _best_window(scores, window_len)

    return CriticalState(
        state=states[start],
        snapshot=snapshots[start],
        importance=scores,
        window_start=start,
        window_length=window_len,
        trajectory_length=L,
    )


@torch.no_grad()
def fidelity_score(
    env,
    target_policy: ActorCritic,
    mask_net: Optional[MaskNet],
    K: float = 0.1,
    n_trajectories: int = 500,
    d_max: float = 1000.0,
    max_steps: int = 1000,
    device: str = "cpu",
) -> dict:
    """Implements the fidelity-score pipeline from the paper + addendum."""
    log_ratios: List[float] = []

    for _ in range(n_trajectories):
        # Phase 1: clean roll-out + scoring
        clean_states, clean_snaps, scores, clean_actions, clean_rewards = (
            _rollout_with_scores(
                env, target_policy, mask_net, max_steps=max_steps, device=device
            )
        )
        clean_R = float(clean_rewards.sum())
        L = len(clean_states)
        if L == 0:
            continue
        window_len = max(1, int(L * K))
        start = (
            _best_window(scores, window_len)
            if mask_net is not None
            else int(np.random.randint(0, max(1, L - window_len + 1)))
        )

        # Phase 2: replay until ``start``, then random for window_len steps,
        #          then resume policy until episode end.
        obs, _ = env.reset()
        # Try to fast-forward via snapshot; if unavailable, replay actions.
        replayed = False
        if clean_snaps[start] is not None and hasattr(env, "restore"):
            replayed = env.restore(clean_snaps[start])
        if not replayed:
            for t in range(start):
                obs, _, term, trunc, _ = env.step(clean_actions[t])
                if term or trunc:
                    break

        perturbed_R = float(clean_rewards[:start].sum())
        for _ in range(window_len):
            random_action = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(random_action)
            perturbed_R += float(r)
            if term or trunc:
                break
        else:
            done = False
            while not done:
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                action, _, _ = target_policy.act(obs_t, deterministic=False)
                obs, r, term, trunc, _ = env.step(action.squeeze(0).cpu().numpy())
                perturbed_R += float(r)
                done = term or trunc

        d = abs(perturbed_R - clean_R)
        if d <= 0 or d_max <= 0:
            continue
        score = float(np.log(d / d_max) - np.log(window_len / max(L, 1)))
        log_ratios.append(score)

    return {
        "fidelity_mean": float(np.mean(log_ratios)) if log_ratios else 0.0,
        "fidelity_std": float(np.std(log_ratios)) if log_ratios else 0.0,
        "n_valid": len(log_ratios),
        "K": K,
    }
