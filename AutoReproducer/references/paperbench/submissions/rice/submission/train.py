"""RICE training entrypoint — supports the three pipeline stages.

Stage A: ``--stage pretrain``  → pre-train a (sub-optimal) target policy via
         PPO. Per the addendum, we use Stable-Baselines3 ``MlpPolicy``
         defaults to match the paper's experimental setup.

Stage B: ``--stage mask``      → train the mask network (Algorithm 1).

Stage C: ``--stage refine``    → refine the target policy via Algorithm 2
         (RICE) or any of the four baselines (``--method``).

All hyperparameters live in ``configs/default.yaml``.
"""

from __future__ import annotations

import argparse
import os
from copy import deepcopy

import numpy as np
import torch
import yaml

from data import make_env
from model import (
    ActorCritic,
    MaskNet,
    MaskNetworkTrainer,
    RICERefiner,
    PPOFinetune,
    StateMaskR,
    JSRL,
    RandomExplanation,
)
from model.mask_network import MaskTrainConfig
from model.refiner import RefineConfig
from model.baselines import (
    PPOFinetuneConfig,
    StateMaskRConfig,
    RandomExplanationConfig,
    JSRLConfig,
)
from utils import Logger


# ---------------------------------------------------------------------- utils
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_actor_critic(env, hidden=(64, 64)) -> ActorCritic:
    obs_dim = int(np.prod(env.observation_space.shape))
    if hasattr(env.action_space, "shape") and len(env.action_space.shape) > 0:
        action_dim = int(env.action_space.shape[0])
        continuous = True
    else:
        action_dim = int(env.action_space.n)
        continuous = False
    return ActorCritic(obs_dim, action_dim, hidden=hidden, continuous=continuous)


def save_model(model: torch.nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: str) -> torch.nn.Module:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model


# ------------------------------------------------------------------- stage A
def stage_pretrain(cfg: dict, env_name: str, out_dir: str):
    """PPO pre-training. We try Stable-Baselines3 first (matches addendum);
    if SB3 is missing we fall back to a vanilla PPO loop using ``ActorCritic``.
    """
    env = make_env(
        env_name,
        seed=cfg["logging"]["seed"],
        max_episode_steps=cfg["env"]["max_episode_steps"],
    )
    save_path = os.path.join(out_dir, f"pretrain_{env_name}.pt")

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env

        sb_env = make_vec_env(
            lambda: make_env(
                env_name,
                seed=cfg["logging"]["seed"],
                max_episode_steps=cfg["env"]["max_episode_steps"],
            ),
            n_envs=1,
        )
        ppo_cfg = cfg["pretrain"]
        model = PPO(
            ppo_cfg["policy"],
            sb_env,
            learning_rate=ppo_cfg["learning_rate"],
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            n_epochs=ppo_cfg["n_epochs"],
            gamma=ppo_cfg["gamma"],
            gae_lambda=ppo_cfg["gae_lambda"],
            clip_range=ppo_cfg["clip_range"],
            ent_coef=ppo_cfg["ent_coef"],
            vf_coef=ppo_cfg["vf_coef"],
            max_grad_norm=ppo_cfg["max_grad_norm"],
            policy_kwargs={"net_arch": ppo_cfg["policy_kwargs"]["net_arch"]},
            verbose=1,
            seed=cfg["logging"]["seed"],
        )
        model.learn(total_timesteps=ppo_cfg["total_timesteps"])
        model.save(save_path.replace(".pt", ""))
        print(f"[pretrain] saved SB3 PPO to {save_path}")
    except ImportError:
        print("[pretrain] stable-baselines3 unavailable; using internal PPO.")
        from model.refiner import RICERefiner, RefineConfig

        policy = build_actor_critic(env)
        cfg_obj = RefineConfig(
            total_timesteps=cfg["pretrain"]["total_timesteps"],
            reset_probability=0.0,  # plain PPO
            rnd_lambda=0.0,
            learning_rate=cfg["pretrain"]["learning_rate"],
            n_steps=cfg["pretrain"]["n_steps"],
            batch_size=cfg["pretrain"]["batch_size"],
            n_epochs=cfg["pretrain"]["n_epochs"],
            gamma=cfg["pretrain"]["gamma"],
            gae_lambda=cfg["pretrain"]["gae_lambda"],
            clip_range=cfg["pretrain"]["clip_range"],
        )
        refiner = RICERefiner(env, policy, mask_net=None, cfg=cfg_obj)
        refiner.train()
        save_model(policy, save_path)
        print(f"[pretrain] saved internal PPO to {save_path}")


# ------------------------------------------------------------------- stage B
def stage_mask(cfg: dict, env_name: str, out_dir: str):
    env = make_env(
        env_name,
        seed=cfg["logging"]["seed"],
        max_episode_steps=cfg["env"]["max_episode_steps"],
    )
    target_policy = build_actor_critic(env)
    pre_path = os.path.join(out_dir, f"pretrain_{env_name}.pt")
    if os.path.exists(pre_path):
        load_model(target_policy, pre_path)
    elif os.path.exists(pre_path.replace(".pt", ".zip")):
        from stable_baselines3 import PPO

        sb_model = PPO.load(pre_path.replace(".pt", ""))
        # copy weights into our internal ActorCritic
        try:
            sb_state = sb_model.policy.state_dict()
            # weights name mismatch — best effort copy of common layers
            for k_local, k_sb in zip(
                target_policy.state_dict().keys(), sb_state.keys()
            ):
                if target_policy.state_dict()[k_local].shape == sb_state[k_sb].shape:
                    target_policy.state_dict()[k_local].copy_(sb_state[k_sb])
        except Exception as e:
            print(f"[mask] could not copy SB3 weights: {e}")
    else:
        print("[mask] WARNING: no pre-trained policy found, using random init.")

    mc = cfg["mask"]
    mask_cfg = MaskTrainConfig(
        total_timesteps=mc["total_timesteps"],
        alpha=mc["alpha"],
        learning_rate=mc["learning_rate"],
        n_steps=mc["n_steps"],
        batch_size=mc["batch_size"],
        n_epochs=mc["n_epochs"],
        gamma=mc["gamma"],
        gae_lambda=mc["gae_lambda"],
        clip_range=mc["clip_range"],
    )
    trainer = MaskNetworkTrainer(env, target_policy, mask_cfg)
    mask_net = trainer.train()
    save_path = os.path.join(out_dir, f"mask_{env_name}.pt")
    save_model(mask_net, save_path)
    print(f"[mask] saved mask network to {save_path}")


# ------------------------------------------------------------------- stage C
def stage_refine(cfg: dict, env_name: str, method: str, out_dir: str):
    env = make_env(
        env_name,
        seed=cfg["logging"]["seed"],
        max_episode_steps=cfg["env"]["max_episode_steps"],
    )
    policy = build_actor_critic(env)
    pre_path = os.path.join(out_dir, f"pretrain_{env_name}.pt")
    if os.path.exists(pre_path):
        load_model(policy, pre_path)
    else:
        print(
            "[refine] WARNING: no pretrained policy at "
            f"{pre_path}; refining a randomly initialised policy."
        )

    obs_dim = int(np.prod(env.observation_space.shape))
    mask_net = MaskNet(obs_dim)
    mask_path = os.path.join(out_dir, f"mask_{env_name}.pt")
    if os.path.exists(mask_path):
        load_model(mask_net, mask_path)
    else:
        print(f"[refine] WARNING: no mask net at {mask_path}; using untrained one.")

    rc = cfg["refine"]
    refine_cfg = RefineConfig(
        total_timesteps=rc["total_timesteps"],
        reset_probability=rc["reset_probability"],
        beta=rc["beta"],
        rnd_lambda=rc["rnd_lambda"],
        rnd_lr=rc["rnd_lr"],
        rnd_feature_dim=rc["rnd_feature_dim"],
        rnd_hidden=tuple(rc["rnd_hidden"]),
        rnd_normalize_obs=rc["rnd_normalize_obs"],
        rnd_normalize_reward=rc["rnd_normalize_reward"],
        trajectory_length_K=rc["trajectory_length_K"],
        learning_rate=rc["learning_rate"],
        n_steps=rc["n_steps"],
        batch_size=rc["batch_size"],
        n_epochs=rc["n_epochs"],
        gamma=rc["gamma"],
        gae_lambda=rc["gae_lambda"],
        clip_range=rc["clip_range"],
    )

    method = method.lower()
    if method == "rice":
        agent = RICERefiner(env, policy, mask_net, refine_cfg)
    elif method == "ppo_ft":
        ft_cfg = PPOFinetuneConfig(**vars(refine_cfg))
        ft_cfg.reset_probability = 0.0
        ft_cfg.rnd_lambda = 0.0
        ft_cfg.learning_rate = cfg["baselines"]["ppo_ft"]["learning_rate"]
        agent = PPOFinetune(env, policy, ft_cfg)
    elif method == "statemask_r":
        sm_cfg = StateMaskRConfig(**vars(refine_cfg))
        sm_cfg.reset_probability = cfg["baselines"]["statemask_r"]["reset_probability"]
        sm_cfg.rnd_lambda = 0.0
        sm_cfg.learning_rate = cfg["baselines"]["statemask_r"]["learning_rate"]
        agent = StateMaskR(env, policy, mask_net, sm_cfg)
    elif method == "jsrl":
        js_cfg = JSRLConfig(
            total_timesteps=rc["total_timesteps"],
            n_curriculum_stages=cfg["baselines"]["jsrl"]["n_curriculum_stages"],
            rollin_horizon_decay=cfg["baselines"]["jsrl"]["rollin_horizon_decay"],
        )
        agent = JSRL(env, policy, js_cfg)
    elif method == "random_expl":
        rx_cfg = RandomExplanationConfig(**vars(refine_cfg))
        agent = RandomExplanation(env, policy, rx_cfg)
    else:
        raise ValueError(f"Unknown method: {method}")

    refined = agent.train()
    save_path = os.path.join(out_dir, f"refine_{method}_{env_name}.pt")
    save_model(refined, save_path)
    print(f"[refine:{method}] saved refined policy to {save_path}")


# ----------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="RICE training script")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--stage", type=str, required=True, choices=["pretrain", "mask", "refine"]
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Override env from config (e.g. Hopper-v4).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="rice",
        help="Refining method: rice / ppo_ft / statemask_r / jsrl / random_expl",
    )
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["logging"]["seed"])
    env_name = args.env or cfg["env"]["name"]
    os.makedirs(args.out_dir, exist_ok=True)

    if args.stage == "pretrain":
        stage_pretrain(cfg, env_name, args.out_dir)
    elif args.stage == "mask":
        stage_mask(cfg, env_name, args.out_dir)
    elif args.stage == "refine":
        stage_refine(cfg, env_name, args.method, args.out_dir)


if __name__ == "__main__":
    main()
