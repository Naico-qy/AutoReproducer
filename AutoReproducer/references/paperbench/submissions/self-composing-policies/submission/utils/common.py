"""Common helpers: config loading, seeding, actor building."""

from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml

from model.architecture import (
    AtariEncoder,
    BaselineActor,
    CompoNetActor,
    PackNet,
    ProgressiveNet,
)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return name


def build_actor(
    method: str, sequence: str, cfg: Dict[str, Any], total_tasks: int
) -> torch.nn.Module:
    """Instantiate the actor for a (method, sequence) pair.

    Mirrors Section 5.2 / Appendix E:
      - For ALE sequences, an Atari CNN encoder is used regardless of method.
      - For Meta-World, the encoder is identity (state is a 39-D vector).
      - The CRL method is applied only to the actor (critic is separate).
    """
    is_atari = sequence in ("spaceinvaders", "freeway")
    if is_atari:
        d_enc = cfg["ale"]["d_enc"]
        n_actions = 6 if sequence == "spaceinvaders" else 3
        encoder = AtariEncoder(
            in_channels=cfg["ale"]["encoder"]["in_channels"],
            d_enc=d_enc,
            image_size=cfg["ale"]["encoder"]["image_size"],
        )
        action_space = "discrete"
    else:
        d_enc = cfg["metaworld"]["d_enc"]
        n_actions = cfg["metaworld"]["n_actions"]
        encoder = torch.nn.Identity()
        action_space = "continuous"

    d_model = cfg["componet"]["d_model"]
    internal_layers = cfg["componet"]["internal_layers"]
    max_modules = cfg["componet"]["max_modules"]

    method = method.lower()
    if method == "componet":
        actor = CompoNetActor(
            encoder=encoder,
            d_enc=d_enc,
            d_model=d_model,
            n_actions=n_actions,
            action_space=action_space,
            internal_layers=internal_layers,
            max_modules=max_modules,
        )
        return actor
    if method == "prognet":
        # ProgressiveNet: a column per task (Rusu et al. 2016).
        # Encoder is shared; pass d_enc as the column input.
        return torch.nn.ModuleDict(
            {
                "encoder": encoder,
                "trunk": ProgressiveNet(
                    d_enc=d_enc,
                    d_model=d_model,
                    n_actions=n_actions,
                    num_layers=cfg.get("metaworld", {}).get("num_actor_layers", 3),
                    action_space=action_space,
                ),
            }
        )
    if method == "packnet":
        return torch.nn.ModuleDict(
            {
                "encoder": encoder,
                "trunk": PackNet(
                    d_enc=d_enc,
                    d_model=d_model,
                    n_actions=n_actions,
                    total_tasks=total_tasks,
                    num_layers=cfg.get("metaworld", {}).get("num_actor_layers", 3),
                    action_space=action_space,
                ),
            }
        )
    if method in ("baseline", "ft1", "ftn"):
        return torch.nn.ModuleDict(
            {
                "encoder": encoder,
                "trunk": BaselineActor(
                    d_enc=d_enc,
                    d_model=d_model,
                    n_actions=n_actions,
                    num_layers=cfg.get("metaworld", {}).get("num_actor_layers", 3),
                    action_space=action_space,
                ),
            }
        )
    raise ValueError(f"Unknown method: {method}")
