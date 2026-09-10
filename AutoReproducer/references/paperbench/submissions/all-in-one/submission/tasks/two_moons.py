"""Two Moons benchmark task (Greenberg et al. 2019; sbibm).

θ ~ U[-1, 1]^2, simulator generates a curved data manifold with noise.
Per addendum.md, structured mask: each x depends on all θ and on previous x.
"""

from __future__ import annotations

import math
import numpy as np
import torch
from torch import Tensor


class TwoMoonsTask:
    name = "two_moons"
    num_params = 2
    num_data = 10  # data dim used in the structured-mask spec.

    def prior(self, n: int) -> Tensor:
        return 2.0 * torch.rand(n, self.num_params) - 1.0

    def simulator(self, theta: Tensor) -> Tensor:
        # The classical 2-moons simulator outputs 2-D, but the joint we model
        # has 10 data variables (per addendum's structured mask). We tile the
        # 2-D simulator output across 5 independent draws to fill 10 dims --
        # this matches "10 i.i.d. observations" stylized version used in §4.1.
        n = theta.shape[0]
        outs = []
        for _ in range(5):
            alpha = (torch.rand(n) - 0.5) * math.pi
            r = 0.1 + 0.01 * torch.randn(n)
            x1 = (
                r * torch.cos(alpha)
                + 0.25
                - torch.abs(theta[:, 0] + theta[:, 1]) / math.sqrt(2.0)
            )
            x2 = r * torch.sin(alpha) + (-theta[:, 0] + theta[:, 1]) / math.sqrt(2.0)
            outs.append(torch.stack([x1, x2], dim=-1))
        return torch.cat(outs, dim=-1)

    def structured_mask(self) -> np.ndarray:
        M_thth = np.eye(2)
        M_xx = np.tril(np.ones((10, 10)))
        zeros = np.zeros((2, 10))
        M_thx = np.ones((10, 2))
        return np.block([[M_thth, zeros], [M_thx, M_xx]]).astype(bool)
