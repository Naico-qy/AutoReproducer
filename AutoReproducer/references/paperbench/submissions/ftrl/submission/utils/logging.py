"""Output helpers."""

from __future__ import annotations

import json
import os
from typing import Any, Dict


def dump_metrics(
    metrics: Dict[str, Any], out_dir: str, name: str = "metrics.json"
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True, default=float)
    return path


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)
