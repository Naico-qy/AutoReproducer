"""
State-diversity diagnostics from Sec. 6.4 / Figs. 7-8 of the paper.

Two metrics are reported in the paper:

1. PCA reconstruction error vs. number of components
   -> A more diverse state distribution requires more components to be
      reconstructed accurately. SAPG should yield a *slower* decay than PPO.

2. MLP reconstruction error
   -> A 2-layer MLP (same hidden size for both layers) with ReLU is trained
      with Adam (PyTorch defaults) on an L2 reconstruction objective for
      400k state transitions (per addendum, point 8-9).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn


def pca_reconstruction_curve(states: torch.Tensor, ks: List[int]) -> List[float]:
    """Compute PCA reconstruction error for each k in `ks`.

    states: (N, D) tensor on any device. Returns a list of MSE values.
    """
    X = states.float()
    X = X - X.mean(0, keepdim=True)
    # SVD-based PCA
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    errors: List[float] = []
    for k in ks:
        kk = min(k, Vh.shape[0])
        Vk = Vh[:kk]  # (k, D)
        proj = X @ Vk.T  # (N, k)
        recon = proj @ Vk  # (N, D)
        err = ((X - recon) ** 2).mean().item()
        errors.append(float(err))
    return errors


def mlp_reconstruction_error(
    states: torch.Tensor,
    hidden: int,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 4096,
    device: str = "cpu",
) -> float:
    """Train a 2-layer MLP (hidden -> hidden, ReLU, Adam defaults) on L2
    reconstruction. Returns the final mean reconstruction MSE on the same
    400k transitions (the paper trains and evaluates on the same set; the
    training error is the diversity diagnostic).
    """
    X = states.float().to(device)
    D = X.shape[-1]
    net = nn.Sequential(
        nn.Linear(D, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, D),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = X.shape[0]
    for _ in range(epochs):
        idx = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            b = idx[s : s + batch_size]
            xb = X[b]
            recon = net(xb)
            loss = ((recon - xb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        err = ((net(X) - X) ** 2).mean().item()
    return float(err)
