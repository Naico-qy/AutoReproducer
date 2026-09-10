"""RoboticSequence environment (paper Algorithm 1, App. B.3).

The agent must complete N Meta-World stages **sequentially** in a single
episode:

    hammer  ->  push  ->  peg-unplug-side  ->  push-wall

Definitions (App. B.3):
* Episodes terminate on stage success **or** time limit (200 steps).
* Successful step yields a "remaining" reward of `beta * r * (T - t)` to
  encourage early success (paper §B.3, beta=1.5).
* The normalised timestep `t / T` is appended to the observation so the MDP
  is fully observable (Pardo et al., 2017).
* A separate output head per stage is used inside the policy/Q-net.

If `metaworld` is not importable, we fall back to a synthetic 6-D linear
"reach a goal" continuous-control surrogate per stage so smoke training
remains runnable. Algorithm 1 logic is preserved either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


def _try_make_metaworld(stage_name: str):
    try:
        import metaworld

        ml1 = metaworld.ML1(stage_name + "-v2")
        return ml1.train_classes[stage_name + "-v2"](), ml1.train_tasks
    except Exception:
        return None, None


@dataclass
class RoboticSequenceConfig:
    sequence: List[str] = field(
        default_factory=lambda: ["hammer", "push", "peg-unplug-side", "push-wall"]
    )
    episode_length: int = 200
    beta_terminal_bonus: float = 1.5
    obs_dim: int = 39  # Meta-World standard observation
    action_dim: int = 4
    seed: int = 0


class _SyntheticStage:
    """Cheap 6-D linear surrogate stage when Meta-World isn't importable."""

    def __init__(self, stage_idx: int, obs_dim: int, action_dim: int, seed: int):
        rng = np.random.default_rng(seed * 7 + stage_idx)
        self.target = rng.standard_normal(obs_dim).astype(np.float32)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state = rng.standard_normal(obs_dim).astype(np.float32)

    def reset(self):
        self.state += np.random.randn(self.obs_dim).astype(np.float32) * 0.1
        return self.state.copy()

    def step(self, action):
        d = self.target[: self.action_dim] - action
        self.state[: self.action_dim] = action
        reward = float(-np.linalg.norm(d))
        success = bool(reward > -0.3)
        return self.state.copy(), reward, success


class RoboticSequenceEnv:
    """Implements Algorithm 1 from the paper."""

    def __init__(self, cfg: RoboticSequenceConfig):
        self.cfg = cfg
        self._stages = []
        self._using_metaworld = []
        for k, name in enumerate(cfg.sequence):
            env, tasks = _try_make_metaworld(name)
            if env is None:
                self._stages.append(
                    _SyntheticStage(k, cfg.obs_dim, cfg.action_dim, cfg.seed)
                )
                self._using_metaworld.append(False)
            else:
                env.seed(cfg.seed + k)
                self._stages.append(env)
                self._using_metaworld.append(True)
        self._idx = 0
        self._t = 0
        self.action_space = type(
            "A", (), {"shape": (cfg.action_dim,), "low": -1.0, "high": 1.0}
        )()
        self.observation_space = type("O", (), {"shape": (cfg.obs_dim + 1,)})()
        self.num_stages = len(self._stages)

    def reset(self):
        self._idx = 0
        self._t = 0
        return self._wrap_obs(self._stages[0].reset()), 0

    def step(self, action: np.ndarray):
        cfg = self.cfg
        stage = self._stages[self._idx]
        if self._using_metaworld[self._idx]:
            obs, reward, terminated, truncated, info = stage.step(action)
            success = bool(info.get("success", False))
        else:
            obs, reward, success = stage.step(action)
        self._t += 1

        if success:
            # apply remaining-reward bonus per App. B.3
            remaining = cfg.episode_length - self._t
            reward = cfg.beta_terminal_bonus * reward * max(remaining, 1)
            self._idx += 1
            self._t = 0
            if self._idx >= self.num_stages:
                # terminal: solved all stages
                done_obs = self._wrap_obs(np.zeros(cfg.obs_dim, dtype=np.float32))
                return (
                    done_obs,
                    reward,
                    True,
                    False,
                    {"stages_solved": self.num_stages},
                    self._idx - 1,
                )
            obs = self._stages[self._idx].reset()

        if self._t >= cfg.episode_length:
            return (
                self._wrap_obs(obs),
                reward,
                True,
                False,
                {"stages_solved": self._idx},
                self._idx,
            )
        return self._wrap_obs(obs), reward, False, False, {}, self._idx

    def _wrap_obs(self, obs: np.ndarray) -> np.ndarray:
        # App. B.3: append normalised timestep `t/T` for full observability
        nt = float(self._t) / float(self.cfg.episode_length)
        return np.concatenate(
            [np.asarray(obs, dtype=np.float32), np.array([nt], dtype=np.float32)]
        )
