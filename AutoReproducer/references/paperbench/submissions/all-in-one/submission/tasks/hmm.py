"""HMM benchmark (paper §4.1, addendum.md).

Latent Markov chain θ_1, ..., θ_10 with θ_1 ~ N(0,1), θ_{k+1} = 0.9 θ_k + ν,
ν ~ N(0, 0.5²). Observations x_k = θ_k + η, η ~ N(0, 0.5²) (factorized).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class HMMTask:
    name = "hmm"
    num_params = 10
    num_data = 10
    trans_coef = 0.9
    proc_std = 0.5
    obs_std = 0.5

    def prior(self, n: int) -> Tensor:
        thetas = torch.zeros(n, self.num_params)
        thetas[:, 0] = torch.randn(n)
        for k in range(1, self.num_params):
            thetas[:, k] = self.trans_coef * thetas[
                :, k - 1
            ] + self.proc_std * torch.randn(n)
        return thetas

    def simulator(self, theta: Tensor) -> Tensor:
        return theta + self.obs_std * torch.randn_like(theta)

    def structured_mask(self) -> np.ndarray:
        M_thth = np.eye(10) + np.diag(np.ones(9), k=-1)
        M_xx = np.eye(10)
        zeros = np.zeros((10, 10))
        M_thx = np.eye(10)
        return np.block([[M_thth, zeros], [M_thx, M_xx]]).astype(bool)
