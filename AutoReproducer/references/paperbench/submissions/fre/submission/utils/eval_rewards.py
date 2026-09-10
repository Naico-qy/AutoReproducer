"""Evaluation reward functions for AntMaze, ExORL, and Kitchen.

All numbers and definitions match the addendum to the FRE paper.

Each reward function takes either a state tensor (B, state_dim) or, where the
reward depends on velocity, a (state, next_state) pair, and returns a (B,)
tensor of scalar rewards.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# AntMaze: 5 hand-crafted goal locations (addendum)
# ---------------------------------------------------------------------------
ANTMAZE_GOALS: dict[str, Tuple[float, float]] = {
    "goal-bottom": (28.0, 0.0),
    "goal-left": (0.0, 15.0),
    "goal-top": (35.0, 24.0),
    "goal-center": (12.0, 24.0),
    "goal-right": (33.0, 16.0),
}


def antmaze_goal_reward(
    state: torch.Tensor, goal_xy: Tuple[float, float], threshold: float = 2.0
) -> torch.Tensor:
    """r = 0 if within `threshold` of goal_xy else -1 (App. C.1)."""
    xy = state[..., :2]
    g = torch.as_tensor(goal_xy, dtype=xy.dtype, device=xy.device)
    d = torch.linalg.vector_norm(xy - g, dim=-1)
    return torch.where(d < threshold, torch.zeros_like(d), -torch.ones_like(d))


# ---------------------------------------------------------------------------
# AntMaze directional: dot product against unit target velocity
# ---------------------------------------------------------------------------
ANTMAZE_DIRECTIONS: dict[str, Tuple[float, float]] = {
    "vel_left": (-1.0, 0.0),
    "vel_up": (0.0, 1.0),
    "vel_down": (0.0, -1.0),
    "vel_right": (1.0, 0.0),
}


def antmaze_directional_reward(
    state: torch.Tensor,
    next_state: torch.Tensor,
    target_dir: Tuple[float, float],
    dt: float = 0.05,
) -> torch.Tensor:
    """Reward = max(0, dot(actual_velocity, unit_target))."""
    vel = (next_state[..., :2] - state[..., :2]) / dt
    t = torch.as_tensor(target_dir, dtype=vel.dtype, device=vel.device)
    t = t / (torch.linalg.vector_norm(t) + 1e-8)
    return torch.clamp((vel * t).sum(-1), min=0.0)


# ---------------------------------------------------------------------------
# Random simplex height map (addendum: opensimplex, seeds 1..5)
# ---------------------------------------------------------------------------
def antmaze_random_simplex_reward(
    state: torch.Tensor, next_state: torch.Tensor, seed: int = 1, scale: float = 0.1
) -> torch.Tensor:
    """Baseline -1 + height bonus + dir bonus.

    Uses opensimplex if available; otherwise a deterministic numpy hash that
    keeps the structure (smooth 2D field).
    """
    xy = state[..., :2].cpu().numpy()
    nxy = next_state[..., :2].cpu().numpy()
    try:
        import opensimplex  # type: ignore

        n = opensimplex.OpenSimplex(seed=seed)
        h = np.array([n.noise2(x * scale, y * scale) for (x, y) in xy])
        # local "preferred direction" is the gradient of the height field
        eps = 1.0
        gx = np.array(
            [
                n.noise2((x + eps) * scale, y * scale)
                - n.noise2((x - eps) * scale, y * scale)
                for (x, y) in xy
            ]
        )
        gy = np.array(
            [
                n.noise2(x * scale, (y + eps) * scale)
                - n.noise2(x * scale, (y - eps) * scale)
                for (x, y) in xy
            ]
        )
    except Exception:  # pragma: no cover
        rng = np.random.default_rng(seed)
        h = rng.standard_normal(xy.shape[0])
        gx = rng.standard_normal(xy.shape[0])
        gy = rng.standard_normal(xy.shape[0])
    vel = nxy - xy
    dir_bonus = vel[:, 0] * gx + vel[:, 1] * gy
    r = -1.0 + h + dir_bonus
    return torch.as_tensor(r, dtype=state.dtype, device=state.device)


# ---------------------------------------------------------------------------
# Hand-crafted path rewards
# ---------------------------------------------------------------------------
def _line_segment_dist(
    xy: torch.Tensor, p1: Tuple[float, float], p2: Tuple[float, float]
) -> torch.Tensor:
    p1 = torch.as_tensor(p1, dtype=xy.dtype, device=xy.device)
    p2 = torch.as_tensor(p2, dtype=xy.dtype, device=xy.device)
    seg = p2 - p1
    t = torch.clamp(((xy - p1) * seg).sum(-1) / (seg.pow(2).sum() + 1e-8), 0.0, 1.0)
    proj = p1 + t.unsqueeze(-1) * seg
    return torch.linalg.vector_norm(xy - proj, dim=-1)


def antmaze_path_center_reward(state: torch.Tensor) -> torch.Tensor:
    xy = state[..., :2]
    d = _line_segment_dist(xy, (0.0, 12.0), (35.0, 12.0))
    return torch.where(d < 3.0, torch.zeros_like(d), -torch.ones_like(d))


def antmaze_path_loop_reward(state: torch.Tensor) -> torch.Tensor:
    xy = state[..., :2]
    centre = torch.tensor([17.5, 12.0], dtype=xy.dtype, device=xy.device)
    radius = 12.0
    d = torch.abs(torch.linalg.vector_norm(xy - centre, dim=-1) - radius)
    return torch.where(d < 2.0, torch.zeros_like(d), -torch.ones_like(d))


def antmaze_path_edges_reward(state: torch.Tensor) -> torch.Tensor:
    xy = state[..., :2]
    d_top = torch.abs(xy[..., 1] - 24.0)
    d_bot = torch.abs(xy[..., 1] - 0.0)
    d_left = torch.abs(xy[..., 0] - 0.0)
    d_right = torch.abs(xy[..., 0] - 35.0)
    d = torch.minimum(torch.minimum(d_top, d_bot), torch.minimum(d_left, d_right))
    return torch.where(d < 2.0, torch.zeros_like(d), -torch.ones_like(d))


# ---------------------------------------------------------------------------
# ExORL velocity rewards
# ---------------------------------------------------------------------------
def cheetah_velocity_reward(
    speed: torch.Tensor, threshold: float, backwards: bool = False
) -> torch.Tensor:
    """Cheetah velocity reward (addendum):

    r = clip(v / threshold, 0, 1) if v in target direction else 0.
    """
    v = -speed if backwards else speed
    return torch.clamp(v / max(threshold, 1e-6), 0.0, 1.0)


def walker_velocity_reward(
    velocity: torch.Tensor, threshold: float, backwards: bool = False
) -> torch.Tensor:
    v = -velocity if backwards else velocity
    return torch.clamp(v / max(threshold, 1e-6), 0.0, 1.0)


# ---------------------------------------------------------------------------
# ExORL goal-reaching (Euclidean threshold 0.1)
# ---------------------------------------------------------------------------
def goal_reaching_reward(
    state: torch.Tensor,
    goal: torch.Tensor,
    threshold: float = 0.1,
    normalizer: torch.Tensor | None = None,
) -> torch.Tensor:
    """r = 0 within threshold else -1.  `normalizer` is per-dim std (App. C.2)."""
    if normalizer is not None:
        s = state / normalizer
        g = goal / normalizer
    else:
        s, g = state, goal
    d = torch.linalg.vector_norm(s - g, dim=-1)
    return torch.where(d < threshold, torch.zeros_like(d), -torch.ones_like(d))
