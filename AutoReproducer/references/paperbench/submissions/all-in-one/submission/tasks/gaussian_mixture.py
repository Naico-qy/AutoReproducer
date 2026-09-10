"""Gaussian Mixture benchmark (Lueckmann et al. 2021).

θ ~ U[-10, 10]^2.  x | θ ~ 0.5 N(θ, I) + 0.5 N(θ, 0.01 I), 10 dims via tile.
We use the same per-row structured mask as Two Moons (per addendum.md, the
two share a mask layout).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class GaussianMixtureTask:
    name = "gaussian_mixture"
    num_params = 2
    num_data = 10

    def prior(self, n: int) -> Tensor:
        return 20.0 * torch.rand(n, self.num_params) - 10.0

    def simulator(self, theta: Tensor) -> Tensor:
        n = theta.shape[0]
        mode = (torch.rand(n, 1) > 0.5).float()
        std = mode * 1.0 + (1.0 - mode) * 0.1
        # Tile to 10 dims (5 i.i.d. 2-D draws)
        outs = []
        for _ in range(5):
            outs.append(theta + std * torch.randn(n, 2))
        return torch.cat(outs, dim=-1)

    def structured_mask(self) -> np.ndarray:
        M_thth = np.eye(2)
        M_xx = np.tril(np.ones((10, 10)))
        zeros = np.zeros((2, 10))
        M_thx = np.ones((10, 2))
        return np.block([[M_thth, zeros], [M_thx, M_xx]]).astype(bool)
