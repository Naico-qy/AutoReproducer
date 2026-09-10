"""SEMA training entry-point.

Implements the continual training loop described in Sec. 3.5 / 3.6:

  for each task t in 1..T:
      if t == 1:
          # First-session adaptation -- train the default adapters/RDs/router
          # at every expansion-enabled layer.
          train adapters + descriptors + router on D^t
      else:
          # Self-expansion scan (Sec. 3.6).
          for layer l in expansion_layers (shallow -> deep):
              compute z-scores from existing RDs at l on the first epoch.
              if all RDs flag the task as novel (z > threshold for a majority
                  of samples at all RDs), expand layer l with a new modular
                  adapter and train it (and the new router column) on D^t.
              else:
                  reuse frozen adapters at l.
          # Always update the prototype classifier on D^t.

After training, prototypes are used by the cosine head for evaluation.

Usage:
    python train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import torch
import yaml
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data import build_continual_dataset
from model import SEMA
from model.architecture import SEMAConfig


# ---------------------------------------------------------------- utilities
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> SEMA:
    sema_cfg = SEMAConfig(
        embed_dim=cfg["embed_dim"],
        num_layers=cfg["num_layers"],
        expansion_layers=cfg["expansion"]["enabled_layers"],
        z_threshold=cfg["expansion"]["z_threshold"],
        adapter_kind=cfg["adapter"]["type"],
        adapter_bottleneck=cfg["adapter"]["bottleneck"],
        descriptor_latent=cfg["descriptor"]["latent_dim"],
        leaky_slope=cfg["descriptor"].get("leaky_slope", 0.2),
        backbone_name=cfg["backbone"],
        pretrained=(cfg["pretrained_weights"] is not None),
    )
    return SEMA(sema_cfg)


# ---------------------------------------------------------------- core ops
def train_first_task(
    model: SEMA,
    loader: DataLoader,
    cfg: dict,
    device: str,
) -> None:
    """First-session training (Appendix A.1): default adapters + RDs + CE."""
    train_params = model.trainable_adapter_parameters()
    desc_params = model.trainable_descriptor_parameters()
    if not train_params and not desc_params:
        return
    opt_a = (
        SGD(
            train_params,
            lr=cfg["training"]["lr_adapter"],
            momentum=cfg["training"]["momentum"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        if train_params
        else None
    )
    opt_d = (
        SGD(
            desc_params,
            lr=cfg["training"]["lr_descriptor"],
            momentum=cfg["training"]["momentum"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        if desc_params
        else None
    )
    sched_a = (
        CosineAnnealingLR(opt_a, T_max=cfg["training"]["epochs_adapter"])
        if opt_a
        else None
    )
    sched_d = (
        CosineAnnealingLR(opt_d, T_max=cfg["training"]["epochs_descriptor"])
        if opt_d
        else None
    )

    ce = nn.CrossEntropyLoss(ignore_index=-1)
    classes = sorted({int(y) for _, y in loader.dataset})
    label_map = {c: i for i, c in enumerate(classes)}

    epochs = max(
        cfg["training"]["epochs_adapter"], cfg["training"]["epochs_descriptor"]
    )
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y_remap = torch.tensor(
                [label_map[int(yi)] for yi in y], device=device, dtype=torch.long
            )

            # Adapter branch (CE on a temporary linear classifier head).
            if opt_a is not None and epoch < cfg["training"]["epochs_adapter"]:
                logits = _temporary_linear_logits(model, x, num_classes=len(classes))
                loss_ce = ce(logits, y_remap)
                opt_a.zero_grad(set_to_none=True)
                loss_ce.backward()
                opt_a.step()

            # Descriptor branch (Eq. 2) -- on the same features.
            if opt_d is not None and epoch < cfg["training"]["epochs_descriptor"]:
                feats = model.extract_layerwise(x)
                loss_rd = sum(_descriptor_loss(model, feats, l) for l in feats.keys())
                opt_d.zero_grad(set_to_none=True)
                loss_rd.backward()
                opt_d.step()

        if sched_a:
            sched_a.step()
        if sched_d:
            sched_d.step()

    _finalise_descriptors(model)
    _accumulate_prototypes(model, loader, device)


def train_subsequent_task(
    model: SEMA,
    loader: DataLoader,
    cfg: dict,
    device: str,
) -> Dict[int, bool]:
    """Self-expansion scan (Sec. 3.6) + selective training."""
    expanded: Dict[int, bool] = {}
    threshold = cfg["expansion"]["z_threshold"]

    # Scan first epoch: for each expansion layer (shallow -> deep) compute
    # z-scores from existing RDs and decide whether to expand.
    model.eval()
    novel_counts: Dict[int, int] = {l: 0 for l in model.expansion_layer_set()}
    sample_count = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            feats = model.extract_layerwise(x)
            sample_count += x.shape[0]
            for l, f in feats.items():
                # z-score: per-sample min over K (Sec. 3.6 -- "all z's > thr").
                z = model.compute_layer_z(f.mean(dim=1) if f.dim() == 3 else f, l)
                novel_counts[l] += int((z > threshold).sum().item())

    # Decide expansion (shallow -> deep): expand if a majority of samples
    # exceed threshold simultaneously across all existing RDs.
    for l in sorted(novel_counts.keys()):
        frac = novel_counts[l] / max(sample_count, 1)
        if frac > 0.5:  # robust aggregation rule
            model.expand_layer(l)
            expanded[l] = True
        else:
            expanded[l] = False

    # If we expanded anything, train just the new slot. Otherwise reuse.
    if any(expanded.values()):
        _train_new_slot(model, loader, cfg, device)
    _accumulate_prototypes(model, loader, device)
    return expanded


# ---------------------------------------------------------------- helpers
def _temporary_linear_logits(
    model: SEMA, x: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Use a hidden running linear head only for the CE step (first task).

    Implementation detail: we instantiate the head lazily as a buffer so it
    survives across epochs of the first task only. This matches the ADAM
    "first-session adaptation" convention.
    """
    feats = _forward_features(model, x)
    if (
        not hasattr(model, "_first_task_head")
        or model._first_task_head.out_features != num_classes
    ):
        model._first_task_head = nn.Linear(
            model.cfg.embed_dim, num_classes, bias=True
        ).to(x.device)
        nn.init.zeros_(model._first_task_head.bias)
    return model._first_task_head(feats)


def _forward_features(model: SEMA, x: torch.Tensor) -> torch.Tensor:
    """Run SEMA forward, return CLS features (B, D)."""
    out = model(x, return_features=True)
    if isinstance(out, tuple):
        feats = out[1] if out[1] is not None else out[0]
    else:
        feats = out
    return feats


def _descriptor_loss(
    model: SEMA, feats: Dict[int, torch.Tensor], l: int
) -> torch.Tensor:
    """Sum of L_RD over currently-trainable descriptors at layer l (Eq. 2)."""
    block = model.get_block(l)
    if block.K == 0:
        return torch.zeros((), device=feats[l].device)
    rd = block.descriptors[-1]  # only the active (last) RD
    if not any(p.requires_grad for p in rd.parameters()):
        return torch.zeros((), device=feats[l].device)
    pooled = feats[l].mean(dim=1) if feats[l].dim() == 3 else feats[l]
    return rd.loss(pooled)


@torch.no_grad()
def _finalise_descriptors(model: SEMA) -> None:
    """After training, finalise running stats to obtain valid sigmas."""
    for block in model.modular_blocks.values():
        for rd in block.descriptors:
            if any(p.requires_grad for p in rd.parameters()):
                rd.finalise_stats()


@torch.no_grad()
def _accumulate_prototypes(model: SEMA, loader: DataLoader, device: str) -> None:
    model.eval()
    classes = sorted({int(y) for _, y in loader.dataset})
    max_class = max(classes)
    if model.prototypes.shape[0] <= max_class:
        old = model.prototypes
        old_count = model.prototype_count
        new = torch.zeros(max_class + 1, model.cfg.embed_dim, device=device)
        new[: old.shape[0]] = old
        cnt = torch.zeros(max_class + 1, dtype=torch.long, device=device)
        cnt[: old_count.shape[0]] = old_count
        model.prototypes = new
        model.prototype_count = cnt
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        feats = _forward_features(model, x)
        model.update_prototypes(feats.float(), y)


def _train_new_slot(model: SEMA, loader: DataLoader, cfg: dict, device: str) -> None:
    train_params = model.trainable_adapter_parameters()
    desc_params = model.trainable_descriptor_parameters()
    if not train_params and not desc_params:
        return
    classes = sorted({int(y) for _, y in loader.dataset})
    label_map = {c: i for i, c in enumerate(classes)}
    opt_a = (
        SGD(
            train_params,
            lr=cfg["training"]["lr_adapter"],
            momentum=cfg["training"]["momentum"],
        )
        if train_params
        else None
    )
    opt_d = (
        SGD(
            desc_params,
            lr=cfg["training"]["lr_descriptor"],
            momentum=cfg["training"]["momentum"],
        )
        if desc_params
        else None
    )
    sched_a = (
        CosineAnnealingLR(opt_a, T_max=cfg["training"]["epochs_adapter"])
        if opt_a
        else None
    )
    sched_d = (
        CosineAnnealingLR(opt_d, T_max=cfg["training"]["epochs_descriptor"])
        if opt_d
        else None
    )
    ce = nn.CrossEntropyLoss()

    epochs = max(
        cfg["training"]["epochs_adapter"], cfg["training"]["epochs_descriptor"]
    )
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y_remap = torch.tensor(
                [label_map[int(yi)] for yi in y], device=device, dtype=torch.long
            )
            if opt_a is not None and epoch < cfg["training"]["epochs_adapter"]:
                logits = _temporary_linear_logits(model, x, num_classes=len(classes))
                loss = ce(logits, y_remap)
                opt_a.zero_grad(set_to_none=True)
                loss.backward()
                opt_a.step()
            if opt_d is not None and epoch < cfg["training"]["epochs_descriptor"]:
                feats = model.extract_layerwise(x)
                loss_rd = sum(_descriptor_loss(model, feats, l) for l in feats.keys())
                opt_d.zero_grad(set_to_none=True)
                loss_rd.backward()
                opt_d.step()
        if sched_a:
            sched_a.step()
        if sched_d:
            sched_d.step()
    _finalise_descriptors(model)


# ---------------------------------------------------------------- main loop
def run(cfg: dict) -> Dict[str, float]:
    device = cfg["training"]["device"] if torch.cuda.is_available() else "cpu"
    seed = cfg["data"]["seed"]
    torch.manual_seed(seed)

    cont = build_continual_dataset(
        name=cfg["data"]["dataset"],
        root=cfg["data"]["data_root"],
        increment=cfg["data"]["increment"],
        init_classes=cfg["data"]["init_classes"],
        seed=seed,
        shuffle_classes=cfg["data"]["shuffle_classes"],
        vtab_order=cfg["data"].get("vtab_order"),
    )

    model = build_model(cfg).to(device)
    model.reset_prototypes(cont.num_classes)

    avg_inc: List[float] = []
    last_acc = 0.0
    for t in range(cont.num_tasks):
        loader = cont.train_loader(
            t,
            batch_size=cfg["training"]["batch_size"],
            num_workers=cfg["training"]["num_workers"],
        )
        t0 = time.time()
        if t == 0:
            train_first_task(model, loader, cfg, device)
        else:
            train_subsequent_task(model, loader, cfg, device)
        # Evaluate on all classes seen so far.
        seen = cont.cumulative_classes(t)
        test_loader = cont.test_loader(
            seen, batch_size=64, num_workers=cfg["training"]["num_workers"]
        )
        acc = evaluate(model, test_loader, device)
        avg_inc.append(acc)
        last_acc = acc
        print(f"[task {t}] acc={acc:.4f}  time={time.time() - t0:.1f}s")

    metrics = {
        "A_N": last_acc,
        "A_bar": float(sum(avg_inc) / max(len(avg_inc), 1)),
        "per_task_acc": avg_inc,
    }
    return metrics


@torch.no_grad()
def evaluate(model: SEMA, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x, return_features=False)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        if logits.shape[-1] == 0:
            continue
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.shape[0])
    return correct / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser(description="SEMA training")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--output", type=str, default="/output/metrics.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    metrics = run(cfg)

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Wrote metrics to", args.output)


if __name__ == "__main__":
    main()
