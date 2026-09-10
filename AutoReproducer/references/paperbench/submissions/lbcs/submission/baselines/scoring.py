"""Coreset selection baselines reimplemented per addendum (§5.2).

For each baseline we expose `*_select(scores, k, ...) -> mask: np.ndarray`
or `*_select(net, dataset, k, ...) -> mask`, as appropriate.

References (verified citations):
  - EL2N / GraNd (Paul et al., 2021, "Deep Learning on a Data Diet", NeurIPS)
        repo: https://github.com/mansheej/data_diet
  - Influential (Yang et al., 2023, ICLR — Dataset Pruning via Influence)
        repo: https://shuoyang-1998.github.io/assets/code/code_datasetptuning.zip
  - Moderate (Xia et al., 2023b, ICLR — "Moderate coreset")
        repo: https://github.com/tmllab/Moderate-DS
  - CCS (Zheng et al., 2023, ICLR — Coverage-Centric Coreset Selection)
        repo: https://github.com/haizhongzheng/Coverage-centric-coreset-selection
  - Probabilistic (Zhou et al., 2022, ICML)
        repo: https://github.com/x-zho14/Probabilistic-Bilevel-Coreset-Selection
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ----------------------- Uniform -----------------------
def uniform_select(n: int, k: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(k, n), replace=False)
    mask = np.zeros(n, dtype=np.uint8)
    mask[idx] = 1
    return mask


# ----------------------- EL2N (Paul et al., 2021) -----------------------
@torch.no_grad()
def el2n_scores(
    net: nn.Module, dataset: Dataset, *, batch_size: int = 256, device: str = "cuda"
) -> np.ndarray:
    """EL2N(x) = || softmax(h(x; theta)) - one_hot(y) ||_2  (Paul et al. 2021, eq. 1)."""
    net.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores = []
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        probs = F.softmax(net(x), dim=1)
        oh = F.one_hot(y, num_classes=probs.size(1)).float()
        s = (probs - oh).norm(p=2, dim=1)
        scores.append(s.detach().cpu().numpy())
    return np.concatenate(scores)


def el2n_select(scores: np.ndarray, k: int) -> np.ndarray:
    """Larger EL2N => harder example => keep top-k."""
    n = scores.shape[0]
    order = np.argsort(-scores)
    mask = np.zeros(n, dtype=np.uint8)
    mask[order[:k]] = 1
    return mask


# ----------------------- GraNd (Paul et al., 2021) -----------------------
def grand_scores(
    net: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int = 1,
    device: str = "cuda",
) -> np.ndarray:
    """GraNd(x) = || grad_theta ell(h(x; theta), y) ||_2 (last-layer approx).

    Following Paul et al. 2021, we approximate by taking the gradient norm of
    the loss w.r.t. the last fully-connected layer; this is the standard
    cheap approximation used in mansheej/data_diet.
    """
    net.to(device).eval()
    # find last linear layer
    last = None
    for m in net.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise RuntimeError("no Linear layer found for GraNd approximation")
    last.weight.requires_grad_(True)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores = []
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        for p in net.parameters():
            if p.grad is not None:
                p.grad.zero_()
        loss = criterion(net(x), y)
        loss.backward()
        if last.weight.grad is None:
            scores.append(np.zeros(x.size(0)))
        else:
            scores.append(np.full(x.size(0), float(last.weight.grad.norm(p=2).item())))
    return np.concatenate(scores)


def grand_select(scores: np.ndarray, k: int) -> np.ndarray:
    """Larger GraNd => more important => keep top-k (Paul et al. 2021)."""
    n = scores.shape[0]
    order = np.argsort(-scores)
    mask = np.zeros(n, dtype=np.uint8)
    mask[order[:k]] = 1
    return mask


# ----------------------- Moderate (Xia et al., 2023b) -----------------------
@torch.no_grad()
def feature_extract(
    net: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int = 256,
    device: str = "cuda",
) -> np.ndarray:
    """Extract penultimate-layer features by hooking the last Linear layer."""
    net.to(device).eval()
    feats = []
    last = None
    for m in net.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise RuntimeError("no Linear layer found for feature extraction")

    captured = {}

    def hook(_, inp, __):
        captured["x"] = inp[0].detach().cpu().numpy()

    h = last.register_forward_hook(hook)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in loader:
        x = batch[0].to(device)
        net(x)
        feats.append(captured["x"])
    h.remove()
    return np.concatenate(feats, axis=0)


def moderate_select(features: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Moderate coreset: keep examples whose distance-to-class-center is closest
    to the per-class median distance (Xia et al., 2023b).
    """
    n = features.shape[0]
    classes = np.unique(labels)
    score = np.zeros(n)
    for c in classes:
        idx = np.where(labels == c)[0]
        center = features[idx].mean(0)
        dist = np.linalg.norm(features[idx] - center, axis=1)
        med = np.median(dist)
        score[idx] = -np.abs(dist - med)  # higher score => closer to median
    order = np.argsort(-score)
    mask = np.zeros(n, dtype=np.uint8)
    mask[order[:k]] = 1
    return mask


# ----------------------- CCS (Zheng et al., 2023) -----------------------
def ccs_select(
    scores: np.ndarray,
    labels: np.ndarray,
    k: int,
    *,
    n_strata: int = 50,
    cutoff: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Coverage-centric Coreset Selection (CCS).

    Algorithm (Zheng et al., 2023):
      1. Drop the highest-`cutoff`-fraction-by-score examples (often hardest /
         noisy).
      2. Stratify the remaining by score into `n_strata` equal-width bins.
      3. Sample the budget budget=k uniformly from the bins (stratified
         sampling across strata).
    """
    rng = np.random.default_rng(seed)
    n = scores.shape[0]
    order = np.argsort(-scores)
    n_drop = int(round(cutoff * n))
    keep = order[n_drop:]
    keep_scores = scores[keep]
    bins = np.linspace(keep_scores.min(), keep_scores.max() + 1e-9, n_strata + 1)
    selected = []
    per_stratum = max(1, k // n_strata)
    for b in range(n_strata):
        in_bin = keep[(keep_scores >= bins[b]) & (keep_scores < bins[b + 1])]
        if len(in_bin) == 0:
            continue
        sample_size = min(per_stratum, len(in_bin))
        chosen = rng.choice(in_bin, size=sample_size, replace=False)
        selected.append(chosen)
    selected = np.concatenate(selected) if selected else np.array([], dtype=np.int64)
    if len(selected) > k:
        selected = rng.choice(selected, size=k, replace=False)
    elif len(selected) < k:
        remaining = np.setdiff1d(keep, selected, assume_unique=False)
        extra = rng.choice(
            remaining, size=min(k - len(selected), len(remaining)), replace=False
        )
        selected = np.concatenate([selected, extra])
    mask = np.zeros(n, dtype=np.uint8)
    mask[selected] = 1
    return mask


# ----------------------- Influential (Yang et al., 2023) -----------------------
def influential_select(
    scores: np.ndarray,
    k: int,
) -> np.ndarray:
    """Influential coreset: select examples with the smallest |influence| (i.e.
    those whose removal least disturbs the validation loss), per Yang et al.
    2023. We accept user-precomputed influence-function scores.
    """
    n = scores.shape[0]
    order = np.argsort(np.abs(scores))
    mask = np.zeros(n, dtype=np.uint8)
    mask[order[:k]] = 1
    return mask


# ----------------------- Probabilistic (Zhou et al., 2022) -----------------------
def probabilistic_select(
    n: int,
    f1_eval: Callable[[np.ndarray], float],
    *,
    k: int,
    T: int = 500,
    C: int = 1,
    lr: float = 2.5,
    seed: int = 0,
) -> np.ndarray:
    """Probabilistic Bilevel Coreset Selection (Zhou et al., 2022).

    Continuous reparameterisation: m_i ~ Bernoulli(s_i). The outer loop uses
    REINFORCE-style policy gradient (eq. 30 of LBCS paper):

        f_1(m) * (m - s) / (s * (1 - s))

    is the per-coordinate stochastic gradient, plus the L0 penalty weighted to
    target a budget of `k` examples (we add a soft budget penalty). The
    objective minimised by Zhou et al. is E[f1(m)] subject to E[||m||_0] ≈ k.
    """
    rng = np.random.default_rng(seed)
    s = np.full(n, k / n, dtype=np.float64)
    for t in range(T):
        # average over C samples
        gs = np.zeros_like(s)
        for _ in range(C):
            m = (rng.random(n) < s).astype(np.uint8)
            f1 = f1_eval(m)
            denom = (s * (1.0 - s)) + 1e-8
            gs += f1 * (m.astype(np.float64) - s) / denom
        gs /= C
        # add a budget regulariser pushing E[||m||_0] = sum(s) toward k
        gs += (s.sum() - k) / max(n, 1)
        s -= lr * gs / np.sqrt(t + 1)  # decay
        s = np.clip(s, 1e-3, 1 - 1e-3)
    # final hard mask: top-k by s
    order = np.argsort(-s)
    mask = np.zeros(n, dtype=np.uint8)
    mask[order[:k]] = 1
    return mask
