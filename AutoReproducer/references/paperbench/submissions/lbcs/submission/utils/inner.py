"""Inner-loop trainer + outer-loop objective evaluator for LBCS.

Implements:
  L(m, theta) = (1 / ||m||_0) * sum_i m_i * ell(h(x_i; theta), y_i)        eq. 1 (paper)
  f1(m)       = (1 / n)     * sum_i              ell(h(x_i; theta(m)), y_i)  eq. 1
  f2(m)       = ||m||_0                                                    eq. 2

`inner_train` minimises L(m, theta) on the masked subset.
`compute_f1`  evaluates f1(m) by averaging cross-entropy on the FULL dataset
              with the trained network.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


def _build_optimizer(
    net: nn.Module, name: str, lr: float, momentum: float, weight_decay: float
):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    raise ValueError(f"unknown optimizer '{name}'")


def inner_train(
    net: nn.Module,
    base_dataset: Dataset,
    mask: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    optimizer: str = "adam",
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    device: str = "cuda",
    scheduler: str = "none",
) -> nn.Module:
    """Train `net` on the masked subset by minimising L(m, theta) (eq. 1).

    Cross-entropy is the default loss (paper notation: ell = cross-entropy).
    """
    indices = np.where(mask == 1)[0].tolist()
    if len(indices) == 0:
        return net  # nothing to train on
    subset = Subset(base_dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=min(batch_size, len(subset)),
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )

    net.to(device).train()
    opt = _build_optimizer(net, optimizer, lr, momentum, weight_decay)
    sched = None
    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for batch in loader:
            x, y = batch[0], batch[1]
            x = x.to(device, non_blocking=True)
            y = torch.as_tensor(y, dtype=torch.long, device=device)
            opt.zero_grad()
            logits = net(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
        if sched is not None:
            sched.step()
    return net


@torch.no_grad()
def compute_f1(
    net: nn.Module,
    full_dataset: Dataset,
    *,
    batch_size: int = 256,
    device: str = "cuda",
) -> float:
    """Compute f1(m) = (1/n) * sum_i ell(h(x_i; theta(m)), y_i)  on full data."""
    loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )
    net.to(device).eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total = 0
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        logits = net(x)
        total_loss += criterion(logits, y).item()
        total += y.numel()
    return total_loss / max(total, 1)


@torch.no_grad()
def compute_accuracy(
    net: nn.Module,
    test_dataset: Dataset,
    *,
    batch_size: int = 256,
    device: str = "cuda",
) -> float:
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    net.to(device).eval()
    correct = 0
    total = 0
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        pred = net(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def f2_value(mask: np.ndarray) -> int:
    """f2(m) = ||m||_0 — eq. 2 of the paper."""
    return int(mask.sum())


def make_outer_evaluator(
    proxy_arch_factory,
    base_dataset: Dataset,
    full_eval_dataset: Dataset,
    *,
    inner_epochs: int,
    batch_size: int,
    lr: float,
    optimizer: str,
    weight_decay: float,
    momentum: float,
    device: str,
    group_size: int = 1,
    n_groups: int | None = None,
    n_total: int | None = None,
    init_state: dict | None = None,
):
    """Build an `evaluate(group_mask) -> (f1, f2)` closure for LexiFlow.

    The outer loop optimizes over a *group-level* mask of length `n_groups`
    (paper §3.2 acceleration trick: examples in the same group share a mask).
    The closure expands the group-mask to the full sample mask of length
    `n_total`, trains the proxy network on the masked subset (warm-started
    from `init_state`), and returns (f1, f2).

    f2 is reported in *example* counts (consistent with paper's Tables 2 & 3).
    """
    if n_total is None:
        n_total = len(base_dataset)
    if n_groups is None:
        n_groups = (n_total + group_size - 1) // group_size

    def expand(group_mask01: np.ndarray) -> np.ndarray:
        if group_size <= 1:
            return group_mask01[:n_total].astype(np.uint8)
        full = np.repeat(group_mask01, group_size)[:n_total].astype(np.uint8)
        return full

    def evaluate(group_mask01: np.ndarray):
        full_mask = expand(group_mask01)
        net = proxy_arch_factory()
        if init_state is not None:
            try:
                net.load_state_dict(init_state, strict=False)
            except Exception:
                pass  # tolerate small architecture variations
        net = inner_train(
            net,
            base_dataset,
            full_mask,
            epochs=inner_epochs,
            batch_size=batch_size,
            lr=lr,
            optimizer=optimizer,
            momentum=momentum,
            weight_decay=weight_decay,
            device=device,
        )
        f1 = compute_f1(net, full_eval_dataset, device=device)
        f2 = f2_value(full_mask)
        return (float(f1), float(f2))

    return evaluate, n_groups
