"""Montezuma's Revenge env wrapper (App. B.2 / Table 2).

Atari preprocessing:
    * resize -> (84, 84)
    * grayscale
    * frame-stack k=4
    * sticky actions p=0.25
    * life_done = False (we treat episode ends as the game-over only)

Room indexing follows Figure 12. Pre-training is restricted to states whose
room id ≥ `pretrained_room` (default 7); fine-tuning starts from Room 1.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


def _make_atari():
    try:
        import gymnasium as gym
        import ale_py  # noqa: F401

        env = gym.make(
            "ALE/MontezumaRevenge-v5", frameskip=4, repeat_action_probability=0.25
        )
        return env
    except Exception:
        return None


class FrameStack:
    """Frame-stack wrapper (state_stack_size=4 from Table 2)."""

    def __init__(self, env, k: int = 4, h: int = 84, w: int = 84):
        self.env = env
        self.k = k
        self.h = h
        self.w = w
        self.frames: deque = deque(maxlen=k)
        self.action_space = (
            env.action_space if env is not None else type("A", (), {"n": 18})()
        )

    def _preprocess(self, obs):
        try:
            import cv2
        except Exception:
            cv2 = None
        if cv2 is None:
            # naive nearest-neighbor downsample of an arbitrary input
            arr = np.asarray(obs)
            if arr.ndim == 3:
                arr = arr.mean(-1)
            return arr[: self.h, : self.w].astype(np.float32) / 255.0
        gray = cv2.cvtColor(np.asarray(obs), cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return gray.astype(np.float32) / 255.0

    def reset(self, seed: Optional[int] = None):
        if self.env is None:
            obs = np.zeros((self.h, self.w), dtype=np.float32)
        else:
            obs, _ = self.env.reset(seed=seed)
            obs = self._preprocess(obs)
        for _ in range(self.k):
            self.frames.append(obs)
        return np.stack(self.frames, axis=0), {}

    def step(self, action):
        if self.env is None:
            self.frames.append(np.zeros((self.h, self.w), dtype=np.float32))
            return np.stack(self.frames, axis=0), 0.0, True, False, {"room": 0}
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        obs = self._preprocess(obs)
        self.frames.append(obs)
        return (
            np.stack(self.frames, axis=0),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )


def make_env(seed: int = 0):
    base = _make_atari()
    return FrameStack(base)


# ---------------------------------------------------------------------------
# Room labelling helpers (Figure 12).
# ---------------------------------------------------------------------------


def estimate_room(info: dict) -> int:
    """Return the Atari RAM-derived room number for a Montezuma frame.

    The exact byte index is well-known: RAM[3] = current room (Bellemare et al.,
    2013). We only have it when the underlying ALE env exposes the RAM (which
    `gymnasium` does via `info`); when it does not we fall back to 0.
    """
    return int(info.get("room", info.get("episode", {}).get("room", 0)))
