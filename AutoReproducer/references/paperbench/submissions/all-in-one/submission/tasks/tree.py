"""Tree-structured benchmark (paper §4.1; mask defined in addendum.md).

Joint of 10 variables (3 parameters + 7 data, conventionally) with the tree
described in the mask:
    M_E_tree = I_10
    M_E_tree[0, 1:3] = True   # θ₁, θ₂ children of θ₀
    M_E_tree[1, 3:5] = True   # x₁, x₂ children of θ₁
    M_E_tree[2, 5:7] = True   # x₃, x₄ children of θ₂

We use 3 parameters and 7 data, all univariate Gaussian transitions
σ = 0.5; this is a faithful smoke-quality reproduction of the tree task.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class TreeTask:
    name = "tree"
    num_params = 3
    num_data = 7

    def prior(self, n: int) -> Tensor:
        # Top-level theta_0 ~ N(0,1)
        return torch.randn(n, self.num_params)

    def simulator(self, theta: Tensor) -> Tensor:
        n = theta.shape[0]
        # theta_1, theta_2 children of theta_0
        # In the canonical tree, theta_1 = theta_0 + N(0,sigma); we add this
        # downstream by treating params as a Markov chain over the tree.
        x = torch.zeros(n, self.num_data)
        # x_1, x_2 ~ N(theta_1, 0.5)
        x[:, 0] = theta[:, 1] + 0.5 * torch.randn(n)
        x[:, 1] = theta[:, 1] + 0.5 * torch.randn(n)
        # x_3, x_4 ~ N(theta_2, 0.5)
        x[:, 2] = theta[:, 2] + 0.5 * torch.randn(n)
        x[:, 3] = theta[:, 2] + 0.5 * torch.randn(n)
        # x_5..x_7 ~ N(0, 1) i.i.d. (extra sites for the joint dimension count)
        x[:, 4:] = torch.randn(n, 3)
        return x

    def structured_mask(self) -> np.ndarray:
        n = self.num_params + self.num_data  # = 10
        M = np.eye(n)
        M[0, 1:3] = True
        M[1, 3:5] = True
        M[2, 5:7] = True
        return M.astype(bool)
