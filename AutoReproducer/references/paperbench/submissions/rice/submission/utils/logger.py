"""Tiny stdout / file logger — keeps the codebase dependency-light."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict


class Logger:
    def __init__(self, log_dir: str = "./runs", run_name: str = "rice"):
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(log_dir, f"{run_name}-{ts}.jsonl")
        self._fh = open(self.path, "w", buffering=1)
        self.run_name = run_name

    def log(self, step: int, **metrics: Any) -> None:
        record: Dict[str, Any] = {"step": step, "run": self.run_name, **metrics}
        self._fh.write(json.dumps(record) + "\n")
        formatted = " | ".join(
            f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}"
            for k, v in record.items()
        )
        print(formatted, flush=True)

    def close(self):
        self._fh.close()
