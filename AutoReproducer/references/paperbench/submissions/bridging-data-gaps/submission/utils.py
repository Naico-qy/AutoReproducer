"""Small utilities: YAML config loading with `extends` support, EMA, seeding."""

from __future__ import annotations

import copy
import json
import os
import random
from typing import Any, Dict

import torch

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


# ---------------------------------------------------------------------
# Config: deep-merge a child YAML into its `extends` parent.
# ---------------------------------------------------------------------
def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("pyyaml is required to load configs")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg, dict) and "extends" in cfg:
        parent_rel = cfg.pop("extends")
        parent_path = os.path.join(os.path.dirname(os.path.abspath(path)), parent_rel)
        parent = load_config(parent_path)
        cfg = _deep_merge(parent, cfg)
    return cfg


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Exponential moving average of model parameters (Ho et al. 2020)
# ---------------------------------------------------------------------
class EMA:
    def __init__(self, params, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow = [p.detach().clone() for p in params]

    @torch.no_grad()
    def update(self, params) -> None:
        for s, p in zip(self.shadow, params):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, params) -> None:
        for s, p in zip(self.shadow, params):
            p.detach().copy_(s)


# ---------------------------------------------------------------------
# Metric IO
# ---------------------------------------------------------------------
def save_metrics(metrics: Dict[str, Any], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
