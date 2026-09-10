"""Episodic memory replay slot for SAC (App. C.3).

The paper keeps 10 % of SAC's replay buffer (= 10 000 transitions out of
100 000) populated with expert transitions from the *pre-training* stages.
That portion is *protected* — never overwritten — so it acts as a permanent
rehearsal pool. New online transitions only overwrite the remaining 90 %.

This class plugs in to a standard ring-buffer replay; sampling is uniform over
all transitions, so 10 % of every batch is in expectation expert data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    stage: int


class EpisodicMemoryBuffer:
    """Ring buffer with a frozen prefix slot for expert transitions."""

    def __init__(
        self, capacity: int, frozen_capacity: int, obs_dim: int, action_dim: int
    ):
        assert frozen_capacity <= capacity, "frozen part cannot exceed total capacity"
        self.capacity = capacity
        self.frozen_capacity = frozen_capacity  # 10000 by default
        self.online_capacity = capacity - frozen_capacity  # 90000

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.stage = np.zeros((capacity,), dtype=np.int64)

        self._frozen_size = 0  # how many expert transitions stored
        self._online_pos = 0  # write head into the online slot
        self._online_size = 0  # how many online transitions stored

    # ----- population ------------------------------------------------------

    def freeze_expert(self, transitions):
        """Pre-populate the frozen slot with expert transitions (called once)."""
        for t in transitions:
            if self._frozen_size >= self.frozen_capacity:
                break
            i = self._frozen_size
            self.obs[i] = t.obs
            self.next_obs[i] = t.next_obs
            self.action[i] = t.action
            self.reward[i] = t.reward
            self.done[i] = float(t.done)
            self.stage[i] = t.stage
            self._frozen_size += 1

    def add(self, t: Transition):
        i = self.frozen_capacity + self._online_pos
        self.obs[i] = t.obs
        self.next_obs[i] = t.next_obs
        self.action[i] = t.action
        self.reward[i] = t.reward
        self.done[i] = float(t.done)
        self.stage[i] = t.stage
        self._online_pos = (self._online_pos + 1) % max(self.online_capacity, 1)
        self._online_size = min(self._online_size + 1, self.online_capacity)

    # ----- sampling --------------------------------------------------------

    def __len__(self) -> int:
        return self._frozen_size + self._online_size

    def sample(self, batch_size: int, device: Optional[str] = None):
        n = len(self)
        if n == 0:
            raise RuntimeError("Episodic memory buffer is empty")
        # uniform over all stored transitions, expert and online alike
        indices = np.random.randint(0, n, size=batch_size)
        # Map online indices back into [frozen_capacity, frozen_capacity+online_size)
        out = {
            "obs": self.obs[indices],
            "action": self.action[indices],
            "reward": self.reward[indices],
            "next_obs": self.next_obs[indices],
            "done": self.done[indices],
            "stage": self.stage[indices],
        }
        if device is None:
            return out
        return {k: torch.as_tensor(v, device=device) for k, v in out.items()}
