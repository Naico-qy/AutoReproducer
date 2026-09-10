"""YAML config -> nested SimpleNamespace."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import yaml


def dict_to_namespace(d: Dict[str, Any]) -> SimpleNamespace:
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    return d


def load_config(path: str) -> SimpleNamespace:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return dict_to_namespace(raw)
