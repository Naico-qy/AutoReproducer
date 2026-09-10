"""SLCP (Simple Likelihood, Complex Posterior) benchmark task.

θ ~ U[-3, 3]^4. Each of 4 i.i.d. observations is x_k = mu(θ) + L(θ) eps_k
where mu = (θ_1, θ_2) and L is the Cholesky factor of a 2x2 covariance
parameterized by θ_3, θ_4. The data is 8-dim (4 obs × 2 dims).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from scipy.linalg import block_diag


class SLCPTask:
    name = "slcp"
    num_params = 4
    num_data = 8  # 4 i.i.d. observations × 2 dims

    def prior(self, n: int) -> Tensor:
        return 6.0 * torch.rand(n, self.num_params) - 3.0

    def simulator(self, theta: Tensor) -> Tensor:
        n = theta.shape[0]
        mu = theta[:, :2]  # (n, 2)
        s1 = theta[:, 2] ** 2
        s2 = theta[:, 3] ** 2
        rho = torch.tanh(theta[:, 0] * theta[:, 1])
        # Covariance per element
        cov = torch.zeros(n, 2, 2)
        cov[:, 0, 0] = s1**2 + 1e-3
        cov[:, 1, 1] = s2**2 + 1e-3
        cov[:, 0, 1] = rho * s1 * s2
        cov[:, 1, 0] = cov[:, 0, 1]
        L = torch.linalg.cholesky(cov + 1e-4 * torch.eye(2))
        outs = []
        for _ in range(4):
            eps = torch.randn(n, 2, 1)
            outs.append((mu.unsqueeze(-1) + L @ eps).squeeze(-1))
        return torch.cat(outs, dim=-1)  # (n, 8)

    def structured_mask(self) -> np.ndarray:
        M_thth = np.eye(4)
        M_xx = block_diag(*[np.tril(np.ones((2, 2))) for _ in range(4)])
        zeros = np.zeros((4, 8))
        M_thx = np.ones((8, 4))
        return np.block([[M_thth, zeros], [M_thx, M_xx]]).astype(bool)
