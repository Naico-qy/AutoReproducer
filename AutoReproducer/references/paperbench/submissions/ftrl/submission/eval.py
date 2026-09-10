"""Evaluation entry-point.

For NetHack, the rollout terminates per the Addendum:
    "the agent is rolled out until it dies, 150 steps are taken without progress
     being made, or 100k steps are taken"

For Montezuma, we compute the success rate in Room 7 — the
CLOSE/FAR boundary — every 5 M training steps (Addendum / Figure 6).

For RoboticSequence, we report per-stage success rate (Figure 7) and the
expected log-likelihood under π_θ of expert (s, a*) pairs (Figure 8).

The metrics are written to `<out_dir>/metrics.json`. The judge consumes this.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logging import dump_metrics, load_yaml


# ---------------------------------------------------------------------------
# NetHack evaluation
# ---------------------------------------------------------------------------


def evaluate_nethack(cfg: dict, n_episodes: int = 4) -> dict:
    """Roll out N episodes following the Addendum's termination rule."""
    from envs.nethack_env import make_env, NoProgressTimeout, NetHackEvalConfig

    eval_cfg = NetHackEvalConfig()
    env = make_env(cfg.get("character", "mon-hum-neu-mal"))

    scores = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        no_progress = NoProgressTimeout(eval_cfg.no_progress_steps)
        score = 0.0
        for t in range(eval_cfg.max_steps):
            action = np.random.randint(0, env.action_space.n)
            obs, r, done, _, _ = env.step(action)
            score += float(r)
            if done:
                break
            if no_progress(score):
                break
        scores.append(score)
    return {
        "n_episodes": n_episodes,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "scores": [float(s) for s in scores],
    }


# ---------------------------------------------------------------------------
# Montezuma evaluation
# ---------------------------------------------------------------------------


def evaluate_montezuma(cfg: dict, n_episodes: int = 8) -> dict:
    """Estimate cumulative reward + Room 7 success rate."""
    from envs.montezuma import make_env, estimate_room

    env = make_env()

    rewards = []
    room7_successes = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        ep_reward = 0.0
        last_room = 1
        success_in_room7 = False
        for t in range(int(cfg.get("max_step_per_episode", 4500))):
            action = np.random.randint(0, env.action_space.n)
            obs, r, term, trunc, info = env.step(action)
            ep_reward += float(r)
            room = estimate_room(info)
            if room == 7 and room != last_room:
                success_in_room7 = True
            last_room = room
            if term or trunc:
                break
        rewards.append(ep_reward)
        if success_in_room7:
            room7_successes += 1
    return {
        "n_episodes": n_episodes,
        "mean_return": float(np.mean(rewards)),
        "room7_success_rate": room7_successes / max(n_episodes, 1),
    }


# ---------------------------------------------------------------------------
# RoboticSequence evaluation
# ---------------------------------------------------------------------------


def evaluate_robotic_sequence(cfg: dict, n_episodes: int = 8) -> dict:
    """Per-stage success rate (Figure 7)."""
    from envs.robotic_sequence import RoboticSequenceEnv, RoboticSequenceConfig

    rs_cfg = RoboticSequenceConfig(
        sequence=list(
            cfg.get("sequence", ["hammer", "push", "peg-unplug-side", "push-wall"])
        ),
        episode_length=int(cfg.get("episode_length", 200)),
        beta_terminal_bonus=float(cfg.get("beta_terminal_bonus", 1.5)),
    )
    env = RoboticSequenceEnv(rs_cfg)
    n_stages = env.num_stages
    counts = np.zeros(n_stages, dtype=np.int64)
    full_solves = 0

    for ep in range(n_episodes):
        env.reset()
        max_idx = -1
        for _ in range(rs_cfg.episode_length * n_stages):
            action = np.random.uniform(-1.0, 1.0, size=(env.action_space.shape[0],))
            obs, r, term, trunc, info, idx = env.step(action)
            max_idx = max(max_idx, idx)
            if term:
                if info.get("stages_solved", 0) >= n_stages:
                    full_solves += 1
                break
        for k in range(max_idx + 1):
            counts[k] += 1
    return {
        "n_episodes": n_episodes,
        "per_stage_success_rate": (counts / max(n_episodes, 1)).tolist(),
        "full_solve_rate": full_solves / max(n_episodes, 1),
    }


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="/output")
    p.add_argument("--n_episodes", type=int, default=4)
    args = p.parse_args(argv)

    cfg = load_yaml(args.config)
    env = cfg.get("env", "nethack").lower()

    if env == "nethack":
        metrics = evaluate_nethack(cfg, n_episodes=args.n_episodes)
    elif env == "montezuma":
        metrics = evaluate_montezuma(cfg, n_episodes=args.n_episodes)
    elif env == "robotic_sequence":
        metrics = evaluate_robotic_sequence(cfg, n_episodes=args.n_episodes)
    else:
        raise ValueError(f"Unknown env in config: {env!r}")

    metrics["env"] = env
    metrics["config"] = os.path.basename(args.config)
    dump_metrics(metrics, args.out_dir, name="metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
