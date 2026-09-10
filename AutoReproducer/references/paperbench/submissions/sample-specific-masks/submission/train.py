"""
SMM training entrypoint.

Implements Algorithm 1 ("Visual Reprogramming with SMM") from
Cai et al. ICML 2024 -- iterative joint update of the shared noise pattern
delta and the mask-generator parameters phi, with iterative label mapping
(ILM) refresh once per epoch.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --dataset cifar10 --network resnet18
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

from data.loader import build_dataloaders
from model.architecture import SMM
from model.label_mapping import build_label_mapping


# ----------------------------------------------------------------------
# Config / utilities
# ----------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply non-None CLI arguments on top of the YAML config."""
    overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    cfg.update(overrides)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--network", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--method", type=str, default=None)
    p.add_argument("--mapping_method", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr_pattern", type=float, default=None)
    p.add_argument("--lr_mask", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--smoke_test",
        action="store_true",
        help="Short run for sanity check / reproduce.sh",
    )
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()


# ----------------------------------------------------------------------
# Train / eval loops
# ----------------------------------------------------------------------
def evaluate(model: SMM, label_map: nn.Module, loader, device: torch.device) -> float:
    """Top-1 accuracy on the test set."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits_p = model(x)  # |Y^P|
            logits_t = label_map(logits_p)  # |Y^T|
            pred = logits_t.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def train_one_epoch(
    model: SMM,
    label_map: nn.Module,
    loader,
    optimizer,
    device: torch.device,
    log_interval: int = 50,
) -> float:
    """One pass over the training set; returns average loss."""
    model.train()
    # Make sure the frozen f_P stays in eval mode (BatchNorm running stats).
    model.classifier.eval()

    loss_meter = 0.0
    n = 0
    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits_p = model(x)
        logits_t = label_map(logits_p)
        loss = F.cross_entropy(logits_t, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_meter += loss.item() * x.size(0)
        n += x.size(0)
        if step % log_interval == 0:
            print(f"  [step {step:5d}/{len(loader)}] loss={loss.item():.4f}")
    return loss_meter / max(n, 1)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args)

    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Data ----------
    bs = int(cfg["batch_size"])
    if cfg["dataset"].lower() in ("dtd", "oxfordpets"):
        bs = int(cfg.get("batch_size_small", 64))
    train_loader, test_loader, num_classes = build_dataloaders(
        name=cfg["dataset"],
        root=cfg["data_root"],
        network=cfg["network"],
        batch_size=bs,
        num_workers=int(cfg.get("num_workers", 4)),
        seed=int(cfg["seed"]),
    )

    # ---------- Model (Eq. 3) ----------
    model = SMM(
        network=cfg["network"],
        method=cfg["method"],
        num_pool_layers=int(cfg["mask_num_pool_layers"]),
    ).to(device)
    print(
        f"[SMM] network={cfg['network']} method={cfg['method']} "
        f"trainable_params={model.num_trainable():,}"
    )

    # ---------- Output mapping f_out ----------
    label_map = build_label_mapping(
        num_target=num_classes,
        mapping_method=cfg["mapping_method"],
        seed=int(cfg["seed"]),
    ).to(device)
    if cfg["mapping_method"] in ("flm", "ilm"):
        # Initialise with frequencies under the *current* (untrained) f_in
        # (Algorithms 3/4 step "compute frequency distribution").
        label_map.frequent_mapping(model, train_loader, device)

    # ---------- Optimiser & schedule (Algorithm 1, Table 9) ----------
    params = [
        {"params": [model.delta], "lr": float(cfg["lr_pattern"])},
    ]
    if model.fmask is not None:
        params.append(
            {"params": list(model.fmask.parameters()), "lr": float(cfg["lr_mask"])}
        )

    optim_name = cfg.get("optimizer", "sgd").lower()
    if optim_name == "sgd":
        optimizer = SGD(
            params,
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
        )
    else:
        optimizer = Adam(params, weight_decay=float(cfg.get("weight_decay", 0.0)))

    scheduler = MultiStepLR(
        optimizer,
        milestones=[int(m) for m in cfg["milestones"]],
        gamma=float(cfg["lr_decay"]),
    )

    # ---------- Train (Algorithm 1) ----------
    epochs = int(cfg["epochs"])
    if cfg.get("smoke_test", False):
        epochs = int(cfg.get("smoke_epochs", 2))
        print(f"[SMOKE] running for {epochs} epoch(s) only")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg.get("ckpt_dir", out_dir / "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "test_acc": [], "epochs_run": 0}
    best_acc = 0.0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        # Algorithm 4: refresh ILM mapping each epoch using current f_in.
        if cfg["mapping_method"] == "ilm":
            label_map.iterative_mapping(model, train_loader, device)
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        loss = train_one_epoch(
            model,
            label_map,
            train_loader,
            optimizer,
            device,
            log_interval=int(cfg.get("log_interval", 50)),
        )
        scheduler.step()
        acc = evaluate(model, label_map, test_loader, device)
        history["train_loss"].append(loss)
        history["test_acc"].append(acc)
        history["epochs_run"] = epoch
        print(f"  -> train_loss={loss:.4f}  test_acc={acc * 100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "delta": model.delta.detach().cpu(),
                    "fmask": (
                        model.fmask.state_dict() if model.fmask is not None else None
                    ),
                    "y_sub": label_map.y_sub.detach().cpu(),
                    "config": cfg,
                    "epoch": epoch,
                    "test_acc": acc,
                },
                ckpt_dir / "best.pt",
            )

    elapsed = time.time() - t0
    print(f"\n[done] best test acc = {best_acc * 100:.2f}%  ({elapsed / 60:.1f} min)")

    # ---------- Persist metrics ----------
    metrics = {
        "dataset": cfg["dataset"],
        "network": cfg["network"],
        "method": cfg["method"],
        "mapping_method": cfg["mapping_method"],
        "best_test_acc": best_acc,
        "final_test_acc": history["test_acc"][-1] if history["test_acc"] else None,
        "epochs_run": history["epochs_run"],
        "trainable_params": model.num_trainable(),
        "history": history,
        "elapsed_seconds": elapsed,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    # Also write to /output/metrics.json (paperbench convention).
    pb_out = Path("/output")
    if pb_out.exists() or os.environ.get("PB_REPRODUCE", "0") == "1":
        try:
            pb_out.mkdir(parents=True, exist_ok=True)
            with open(pb_out / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] could not write /output/metrics.json: {e}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
