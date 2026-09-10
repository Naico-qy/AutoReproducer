"""AppleRetrieval toy gridworld (App. A.2).

A 1-D environment with two phases:

    Phase 1 (start at x=0): go to x=M to fetch the apple.
    Phase 2 (after apple)   : return to x=0.

The observation encodes the phase: o = [-c] in Phase 1, o = [+c] in Phase 2.
The optimal policy is "go right in Phase 1, left in Phase 2".

The episode terminates on success or after 100 steps. We use this for the
linear-policy demonstration of FPC (App. A.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AppleRetrievalConfig:
    M: int = 30
    c: float = 1.0
    horizon: int = 100


class AppleRetrieval:
    def __init__(self, cfg: AppleRetrievalConfig):
        self.cfg = cfg
        self._x = 0
        self._phase = 1
        self._t = 0

    @property
    def num_actions(self) -> int:
        return 2  # 0 -> left, 1 -> right

    def reset(self):
        self._x = 0
        self._phase = 1
        self._t = 0
        return self._obs()

    def step(self, action: int):
        # 0 = left (-1), 1 = right (+1)
        delta = 1 if action == 1 else -1
        target_dir = +1 if self._phase == 1 else -1
        reward = +1.0 if delta == target_dir else -1.0
        self._x = max(min(self._x + delta, self.cfg.M), 0)
        self._t += 1

        # Phase transitions
        if self._phase == 1 and self._x >= self.cfg.M:
            self._phase = 2
        done = bool(self._phase == 2 and self._x <= 0) or self._t >= self.cfg.horizon
        return self._obs(), reward, done, {"phase": self._phase, "x": self._x}

    def _obs(self):
        sign = -1.0 if self._phase == 1 else 1.0
        return np.array([sign * self.cfg.c], dtype=np.float32)
