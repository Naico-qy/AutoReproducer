"""Gaussian Linear benchmark task (Lueckmann et al. 2021 / sbibm).

θ ~ N(0, I_10),  x | θ ~ N(θ, 0.1 I_10).
The directed mask is given in addendum.md (per-dimension factorized).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class GaussianLinearTask:
    name = "gaussian_linear"
    num_params = 10
    num_data = 10
    obs_std = 0.1

    def prior(self, n: int) -> Tensor:
        return torch.randn(n, self.num_params)

    def simulator(self, theta: Tensor) -> Tensor:
        return theta + self.obs_std * torch.randn_like(theta)

    def structured_mask(self) -> np.ndarray:
        """Directed adjacency, addendum.md."""
        M_thth = np.eye(10)
        M_xx = np.eye(10)
        zeros = np.zeros((10, 10))
        M_thx = np.eye(10)
        return np.block([[M_thth, zeros], [M_thx, M_xx]]).astype(bool)

    def reference_posterior(self, x_obs: Tensor, n_samples: int) -> Tensor:
        """Closed-form posterior  N( x/(1+0.01) , 0.01/(1.01) I ).

        Used as ground truth for the C2ST evaluation.
        """
        prior_var = 1.0
        like_var = self.obs_std**2
        post_var = 1.0 / (1.0 / prior_var + 1.0 / like_var)
        post_mean = post_var * (x_obs / like_var)  # mu_0 = 0
        return post_mean + np.sqrt(post_var) * torch.randn(n_samples, self.num_params)
