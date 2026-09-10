"""Tiny logger helpers."""

from __future__ import annotations

import logging
import os
import sys


def setup_logger(
    name: str = "stochastic_interpolants", level: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def format_step(step: int, total: int, loss: float, lr: float) -> str:
    return f"step {step:>7}/{total:<7}  loss={loss:+.6f}  lr={lr:.2e}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
