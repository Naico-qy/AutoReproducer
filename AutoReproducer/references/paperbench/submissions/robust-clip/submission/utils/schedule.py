"""Learning-rate schedule used by the paper (Sec. B.1).

Cosine decay with linear warmup, where the warmup spans the first 7% of total
training steps and reaches a peak LR (1e-5 in the chosen setting).
"""

from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


def cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int
):
    """Returns a LambdaLR scheduler implementing the paper's schedule.

    The linear warmup ramps from 0 to peak LR at `num_warmup_steps`. After
    that, the cosine half-period decays the LR to 0 by `num_training_steps`.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / max(1.0, float(num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / max(
            1.0, float(num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
