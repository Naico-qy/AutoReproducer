"""Fisher diagonal estimator for EWC (Kirkpatrick et al., 2017).

Per the Addendum: *"To compute the Fisher matrix 10000 batches should be
sampled from the NLD-AA dataset."* This module implements that procedure for
both NetHack (categorical policy on NLD-AA) and SAC (Gaussian policy on the
expert replay buffer).

For a categorical policy `π_θ(a|s)`, the empirical Fisher is

    F_i  =  E_{s, a ∼ π}  [ ( ∂ log π_θ(a|s) / ∂θ_i )^2 ]

We follow Wolczyk et al. (2022, App. C) and use samples from the *expert*
trajectories (the same buffer as BC) — this is what the Addendum points to.
For the SAC case the Fisher is computed on the squared gradient of the log
density of the actor, restricted to the actor params (App. C.5).
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable

import torch


def estimate_fisher_diagonal(
    model: torch.nn.Module,
    sample_batches: Iterable,
    log_prob_fn: Callable,
    n_batches: int = 10000,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Compute the diagonal of the Fisher information matrix.

    Args
    ----
    model : the network whose parameters we evaluate the Fisher on.
    sample_batches : an iterable yielding batches `b` (any structure).
    log_prob_fn : callable `(model, b) -> log_prob_tensor` returning a
        sum-reducible tensor of log-probabilities (or log-densities) of the
        sampled actions under the current model.
    n_batches : number of mini-batches to average over (10 000 per Addendum).
    device : torch device.

    Returns
    -------
    Dict mapping parameter name -> Fisher diagonal tensor (same shape as the
    parameter).
    """
    model = model.to(device).eval()
    fisher = {
        n: torch.zeros_like(p, device=device)
        for n, p in model.named_parameters()
        if p.requires_grad
    }

    it = iter(sample_batches)
    counted = 0
    for step in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(sample_batches)
            batch = next(it)

        model.zero_grad(set_to_none=True)
        logp = log_prob_fn(model, batch).sum()
        logp.backward()

        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            fisher[name] += p.grad.detach().pow(2)
        counted += 1

    if counted == 0:
        return fisher
    for k in fisher:
        fisher[k] /= float(counted)
    return fisher


def save_fisher(fisher: Dict[str, torch.Tensor], path: str) -> None:
    torch.save(fisher, path)


def load_fisher(path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in torch.load(path, map_location=device).items()}
