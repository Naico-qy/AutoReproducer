"""NetHack environment wrapper.

Per Addendum, the canonical environment is `nle` (https://github.com/heiner/nle).
We import lazily so the rest of the code remains usable on machines without
the C++ NLE dependency. When `nle` is installed, `make_env(...)` constructs a
`gym.make("NetHackChallenge-v0")` with the Human Monk character.

Evaluation termination rule (Addendum):
    "the agent is rolled out until it dies, 150 steps are taken without progress
     being made, or 100k steps are taken"

`NoProgressTimeout` enforces the 150-step no-progress condition by tracking
the in-game score; `MaxSteps(100_000)` enforces the absolute cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class NetHackEvalConfig:
    no_progress_steps: int = 150
    max_steps: int = 100_000


def _ensure_nle():
    try:
        import nle  # noqa: F401
        import gymnasium as gym  # noqa: F401

        return True
    except Exception:
        return False


def make_env(character: str = "mon-hum-neu-mal", seed: int = 0):
    """Build the NLE environment for fine-tuning.

    Returns the raw gym env (action space size 121).  If `nle` is not
    available, returns a small `_FakeNetHack` so that smoke training does not
    crash.
    """
    if not _ensure_nle():
        return _FakeNetHack(seed=seed)
    import nle  # noqa: F401
    import gymnasium as gym

    env = gym.make("NetHackChallenge-v0", character=character)
    env.reset(seed=seed)
    return env


class _FakeNetHack:
    """Lightweight stand-in environment used when `nle` cannot be imported.

    Produces observations with the same shape as the real NLE so smoke
    training pipelines run without the C++ dependency.
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.action_space = type("A", (), {"n": 121})()
        self._t = 0

    def reset(self, seed: Optional[int] = None, **kwargs):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        reward = float(self.rng.normal())
        done = bool(self._t >= 200)
        return self._obs(), reward, done, False, {}

    def _obs(self):
        return {
            "chars": self.rng.integers(0, 256, size=(21, 79), dtype=np.int64),
            "colors": self.rng.integers(0, 16, size=(21, 79), dtype=np.int64),
            "blstats": self.rng.normal(size=(27,)).astype(np.float32),
            "message": self.rng.normal(size=(256,)).astype(np.float32),
        }


class NoProgressTimeout:
    """Tracks lack-of-score-progress per Addendum (150 steps)."""

    def __init__(self, no_progress_steps: int = 150):
        self.no_progress_steps = no_progress_steps
        self._best_score = -np.inf
        self._stagnation = 0

    def __call__(self, score: float) -> bool:
        if score > self._best_score:
            self._best_score = score
            self._stagnation = 0
            return False
        self._stagnation += 1
        return self._stagnation >= self.no_progress_steps

    def reset(self):
        self._best_score = -np.inf
        self._stagnation = 0
