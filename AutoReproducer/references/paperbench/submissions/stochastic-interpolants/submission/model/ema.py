"""Exponential Moving Average of model parameters.

EMA is standard in stochastic-interpolant / diffusion training; the
upstream lucidrains repository keeps an EMA copy with decay 0.9999 by
default. We mirror that here.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import torch
import torch.nn as nn


class EMA:
    """Polyak-Ruppert / EMA averaged copy of a `nn.Module`.

    Usage
    -----
    >>> ema = EMA(model, decay=0.9999)
    >>> ...
    >>> # after every optimiser.step():
    >>> ema.update(model)
    >>> # for sampling:
    >>> ema.copy_to(model)            # or use ema.module directly
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        self.module = deepcopy(model)
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.module.eval()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        esd = self.module.state_dict()
        for k, v in msd.items():
            if v.dtype.is_floating_point:
                esd[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                esd[k].copy_(v)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.module.state_dict())

    def state_dict(self):
        return {"decay": self.decay, "module": self.module.state_dict()}

    def load_state_dict(self, sd):
        self.decay = sd.get("decay", self.decay)
        self.module.load_state_dict(sd["module"])
