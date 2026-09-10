"""LBCS — main training entrypoint.

Algorithm 1 (paper, §3.2):
    Require: a network theta, a dataset D, predefined size k, compromise eps;
    1. Initialize masks m randomly with ||m||_0 = k;
    2. for outer iter t = 1..T:
           Train inner loop: theta(m) <- argmin_theta L(m, theta)            (eq. 1 inner)
           Update masks m via lexicographic optimization (Algorithm 2)        (Appendix A)
    3. Return final mask m.

This script:
    (1) loads dataset (with optional label noise / class imbalance, §5.3),
    (2) optionally injects symmetric label noise (Ma et al. 2020),
    (3) initialises mask with ||m||_0 = k (random),
    (4) runs LexiFlow with the proxy network as the inner loop,
    (5) saves the optimized mask, then trains a fresh `target_arch` on the
        coreset (with appropriate optimizer / scheduler from §5.2),
    (6) writes /output/metrics.json.

Usage:
    python train.py --config configs/default.yaml \
        [--dataset fmnist|svhn|cifar10|mnist_s] [--k 1000] [--epsilon 0.2] \
        [--T 500] [--noise-rate 0.0] [--imbalance-ratio 0.0] [--output-dir ./out]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from data.loader import (
    SubsetDataset,
    get_dataset,
    inject_symmetric_noise,
    make_class_imbalanced,
)
from model.architecture import build_model
from utils.inner import compute_accuracy, inner_train, make_outer_evaluator
from utils.lexiflow import lexiflow_search


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument(
        "--k", type=int, default=None, help="predefined coreset size (initial ||m||_0)"
    )
    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--noise-rate", type=float, default=None)
    p.add_argument("--imbalance-ratio", type=float, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="quick smoke run (small T, few epochs) for CI / reproduce.sh",
    )
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    cfg = deepcopy(cfg)
    if args.dataset:
        cfg["dataset"]["name"] = args.dataset
    if args.k is not None:
        cfg["coreset_train"]["k"] = args.k
    if args.epsilon is not None:
        cfg["outer"]["epsilon"] = args.epsilon
    if args.T is not None:
        cfg["outer"]["T"] = args.T
    if args.noise_rate is not None:
        cfg["dataset"]["noise_rate"] = args.noise_rate
    if args.imbalance_ratio is not None:
        cfg["dataset"]["imbalance_ratio"] = args.imbalance_ratio
    if args.output_dir:
        cfg["experiment"]["output_dir"] = args.output_dir
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    if args.data_root:
        cfg["dataset"]["data_root"] = args.data_root
    if args.smoke:
        cfg["outer"]["T"] = 5
        cfg["inner"]["epochs"] = 1
        cfg["outer"]["inner_epochs"] = 1
        cfg["coreset_train"]["epochs"] = 2
        cfg["outer"]["group_size"] = max(cfg["outer"].get("group_size", 1), 1)
    return cfg


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_mask(n: int, k: int, seed: int) -> np.ndarray:
    """Random binary mask with exactly ||m||_0 = k (Step 2 of Algorithm 1)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(k, n), replace=False)
    mask = np.zeros(n, dtype=np.uint8)
    mask[idx] = 1
    return mask


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_overrides(cfg, args)

    set_seed(int(cfg["experiment"]["seed"]))

    out_dir = cfg["experiment"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[lbcs] device={device}")

    # 1) Dataset
    ds_name = cfg["dataset"]["name"]
    train_set, test_set, num_classes, in_channels = get_dataset(
        ds_name, cfg["dataset"]["data_root"], download=cfg["dataset"]["download"]
    )

    # 2) Imperfect supervision (§5.3)
    nr = float(cfg["dataset"].get("noise_rate", 0.0))
    if nr > 0.0:
        if hasattr(train_set, "targets"):
            new_targets = inject_symmetric_noise(
                train_set.targets, nr, num_classes, seed=cfg["experiment"]["seed"]
            )
            train_set.targets = (
                new_targets.tolist()
                if hasattr(new_targets, "tolist")
                else list(new_targets)
            )
        elif hasattr(train_set, "labels"):
            new_targets = inject_symmetric_noise(
                train_set.labels, nr, num_classes, seed=cfg["experiment"]["seed"]
            )
            train_set.labels = np.asarray(new_targets, dtype=np.int64)
    ir = float(cfg["dataset"].get("imbalance_ratio", 0.0))
    if 0.0 < ir < 1.0:
        if hasattr(train_set, "targets"):
            tgt = np.asarray(train_set.targets)
        elif hasattr(train_set, "labels"):
            tgt = np.asarray(train_set.labels)
        else:
            tgt = np.array([train_set[i][1] for i in range(len(train_set))])
        idx = make_class_imbalanced(
            tgt, ir, num_classes, seed=cfg["experiment"]["seed"]
        )
        train_set = Subset(train_set, idx.tolist())

    n_total = len(train_set)
    print(f"[lbcs] dataset={ds_name} n_total={n_total} num_classes={num_classes}")

    # 3) Initial mask
    k = int(cfg["coreset_train"]["k"])
    seed = int(cfg["experiment"]["seed"])
    init_mask01 = init_mask(n_total, k, seed)

    # 4) LexiFlow inner+outer with proxy network
    proxy_arch = cfg["inner"]["proxy_arch"]

    def proxy_factory():
        return build_model(proxy_arch, num_classes=num_classes, in_channels=in_channels)

    group_size = int(cfg["outer"].get("group_size", 1))
    n_groups = (n_total + group_size - 1) // group_size

    eval_fn, n_groups = make_outer_evaluator(
        proxy_factory,
        base_dataset=train_set,
        full_eval_dataset=train_set,  # paper: f1 evaluated on full data
        inner_epochs=int(cfg["outer"].get("inner_epochs", 1)),
        batch_size=int(cfg["inner"]["batch_size"]),
        lr=float(cfg["inner"]["lr"]),
        optimizer=cfg["inner"]["optimizer"],
        weight_decay=float(cfg["inner"].get("weight_decay", 0.0)),
        momentum=float(cfg["inner"].get("momentum", 0.9)),
        device=device,
        group_size=group_size,
        n_groups=n_groups,
        n_total=n_total,
    )

    # group-level initial mask
    if group_size > 1:
        init_groupmask = np.zeros(n_groups, dtype=np.uint8)
        # mark groups that contain any selected example
        for i in np.where(init_mask01 == 1)[0]:
            init_groupmask[i // group_size] = 1
    else:
        init_groupmask = init_mask01

    print(
        f"[lbcs] starting LexiFlow: T={cfg['outer']['T']}, eps={cfg['outer']['epsilon']}, "
        f"groups={n_groups}, init_size={int(init_groupmask.sum())}"
    )
    t0 = time.time()
    best_groupmask, best_F = lexiflow_search(
        n=n_groups,
        evaluate=eval_fn,
        T=int(cfg["outer"]["T"]),
        epsilon=float(cfg["outer"]["epsilon"]),
        delta_init=float(cfg["outer"]["delta_init"]),
        delta_lower=float(cfg["outer"]["delta_lower"]),
        init_mask01=init_groupmask,
        seed=seed,
        log_every=int(cfg["log"]["interval"]),
    )
    elapsed = time.time() - t0

    # expand back to full-mask
    if group_size > 1:
        best_mask01 = np.repeat(best_groupmask, group_size)[:n_total].astype(np.uint8)
    else:
        best_mask01 = best_groupmask.astype(np.uint8)
    coreset_size = int(best_mask01.sum())
    print(
        f"[lbcs] LexiFlow done in {elapsed:.1f}s. final f1={best_F[0]:.4f} "
        f"f2={int(best_F[1])} (||m||_0 actual={coreset_size})"
    )

    # 5) Save mask
    if cfg["log"].get("save_mask", True):
        np.save(os.path.join(out_dir, "coreset_mask.npy"), best_mask01)

    # 6) Train target network on coreset (§5.2)
    target_arch = cfg["coreset_train"]["target_arch"]
    target_net = build_model(
        target_arch, num_classes=num_classes, in_channels=in_channels
    )
    target_optimizer = cfg["coreset_train"]["optimizer"]
    target_lr = float(cfg["coreset_train"]["lr"])
    target_epochs = int(cfg["coreset_train"]["epochs"])
    target_scheduler = cfg["coreset_train"].get("scheduler", "none")

    target_net = inner_train(
        target_net,
        train_set,
        best_mask01,
        epochs=target_epochs,
        batch_size=int(cfg["coreset_train"]["batch_size"]),
        lr=target_lr,
        optimizer=target_optimizer,
        momentum=float(cfg["coreset_train"]["momentum"]),
        weight_decay=float(cfg["coreset_train"]["weight_decay"]),
        device=device,
        scheduler=target_scheduler,
    )

    test_acc = compute_accuracy(target_net, test_set, device=device)
    print(f"[lbcs] test_acc={test_acc * 100:.2f}%   coreset_size={coreset_size}")

    # 7) Save artefacts + metrics.json (PaperBench reads this)
    metrics = {
        "dataset": ds_name,
        "k_initial": k,
        "epsilon": float(cfg["outer"]["epsilon"]),
        "T": int(cfg["outer"]["T"]),
        "f1_final": float(best_F[0]),
        "f2_final": int(best_F[1]),
        "coreset_size": coreset_size,
        "test_accuracy": float(test_acc),
        "elapsed_seconds": float(elapsed),
        "noise_rate": nr,
        "imbalance_ratio": ir,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(
        {"state_dict": target_net.state_dict(), "metrics": metrics},
        os.path.join(out_dir, "target_model.pt"),
    )
    print(f"[lbcs] wrote metrics to {os.path.join(out_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
