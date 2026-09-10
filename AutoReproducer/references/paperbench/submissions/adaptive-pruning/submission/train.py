"""train.py — main training entrypoint for APT.

Realises the full Algorithm sketched in §4 of:
    Zhao, Hajishirzi & Cao,
    "APT: Adaptive Pruning and Tuning Pretrained Language Models for
     Efficient Training and Inference", ICML 2024.

Key training loop ingredients:
  * Adaptive Pruning  (PruneController) — §4.2
  * Adaptive Tuning   (RankController)  — §4.3
  * Self-Distillation (SelfDistiller)   — §4.4 + Addendum
  * Outlier-aware EMA salience          — Addendum

Usage:
    python train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from data import build_dataloaders, get_task_metric
from model import (
    APTModel,
    PruneController,
    RankController,
    SelfDistiller,
    build_apt_model,
    cofi_distill_loss,
    cubic_sparsity_schedule,
    layer_mapping,
)
from model.distillation import sample_teacher_layers, mu_schedule
from model.salience import EMASalience


# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
def linear_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lam(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return LambdaLR(optimizer, lr_lambda=lam)


# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model: APTModel, loader, device: str) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            token_type_ids=batch.get("token_type_ids"),
            labels=batch.get("labels"),
        )
        preds = out["logits"].argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].numel()
    model.train()
    return {"accuracy": correct / max(1, total)}


# --------------------------------------------------------------------------- #
def teacher_forward(model: APTModel, batch, distiller: SelfDistiller, device: str):
    """Forward through the *teacher* checkpoint.

    The teacher reuses the same frozen base; we only swap-in the cached
    adapter weights for the duration of the forward pass.
    """
    state = distiller.teacher_state()
    if not state:
        # First step — no teacher yet; just use student.
        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                token_type_ids=batch.get("token_type_ids"),
                output_hidden_states=True,
            )
        return out["logits"], out.get("hidden_states", [])

    # Snapshot current student & swap teacher weights in.
    backup = {n: p.detach().clone() for n, p in model.named_parameters() if n in state}
    with torch.no_grad():
        for n, t in state.items():
            if n in dict(model.named_parameters()):
                dict(model.named_parameters())[n].data.copy_(t)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            token_type_ids=batch.get("token_type_ids"),
            output_hidden_states=True,
        )
        # Restore student weights.
        for n, t in backup.items():
            dict(model.named_parameters())[n].data.copy_(t)
    return out["logits"].detach(), [
        h.detach() for h in (out.get("hidden_states", []) or [])
    ]


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max_steps for smoke-quality runs.",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Run a minimal smoke test (50 steps)."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["runtime"]["output_dir"] = args.output_dir
    if args.max_steps:
        cfg["optim"]["max_steps"] = args.max_steps
    if args.smoke:
        cfg["optim"]["max_steps"] = 50
        cfg["pruning"]["pruning_start_step"] = 10
        cfg["pruning"]["pruning_end_step"] = 30
        cfg["tuning"]["rank_growth_steps"] = [15, 25]
        cfg["tuning"]["rank_budget_schedule"] = [1.0, 1.5, 2.0]

    set_seed(cfg["runtime"]["seed"])
    out_dir = Path(cfg["runtime"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[APT] device = {device}")

    # --- 1) Build model + data ---------------------------------------------
    train_loader, eval_loader, num_labels = build_dataloaders(cfg)
    cfg["model"]["num_labels"] = num_labels
    model: APTModel = build_apt_model(cfg).to(device)
    model.enable_tracking(True)
    print(
        f"[APT] backbone = {cfg['model']['name']}, "
        f"trainable = {model.num_trainable():,}, total = {model.num_total():,}"
    )

    # --- 2) Optimizer + LR scheduler ---------------------------------------
    o = cfg["optim"]
    params_adapter = [
        p for p in model.trainable_parameters() if p.dim() > 1
    ]  # adapter matrices
    params_other = [p for p in model.trainable_parameters() if p.dim() <= 1]
    optimizer = AdamW(
        [
            {"params": params_adapter, "lr": o["lr_adapter"]},
            {"params": params_other, "lr": o["lr_classifier"]},
        ],
        betas=tuple(o["betas"]),
        eps=o["eps"],
        weight_decay=o["weight_decay"],
    )
    scheduler = linear_warmup_scheduler(optimizer, o["warmup_steps"], o["max_steps"])

    # --- 3) Controllers ----------------------------------------------------
    p = cfg["pruning"]
    pruner = PruneController(
        target_sparsity=p["target_sparsity"],
        start_step=p["pruning_start_step"],
        end_step=p["pruning_end_step"],
        ema_decay=p["ema_decay"],
        use_kurtosis=p["use_kurtosis"],
        prune_heads=p["prune_components"]["mha_heads"],
        prune_ffn=p["prune_components"]["ffn_neurons"],
        prune_hidden=p["prune_components"]["hidden_dim"],
    )
    pruner.reset(model)

    t = cfg["tuning"]
    ranker = RankController(
        init_rank=cfg["adapter"]["init_rank"],
        max_rank=cfg["adapter"]["max_rank"],
        growth_steps=t["rank_growth_steps"],
        rank_budget_schedule=t["rank_budget_schedule"],
        topk_fraction=t["topk_fraction"],
    )

    d = cfg["distillation"]
    distiller = SelfDistiller(
        hidden_dim=model.hidden_size,
        num_sampled_layers=d["num_teacher_layers"],
        temperature=d["temperature"],
        layer_loss_weight=d["layer_loss_weight"],
        pred_loss_weight=(
            d["pred_loss_weight_classification"]
            if cfg["model"]["task_type"] == "classification"
            else d["pred_loss_weight_generation"]
        ),
    ).to(device)

    # --- 4) Training loop --------------------------------------------------
    metrics_history = []
    step = 0
    t0 = time.time()
    n_layers = getattr(model.backbone.config, "num_hidden_layers", 12)
    rng = random.Random(cfg["runtime"]["seed"])

    train_iter = iter(train_loader)
    while step < o["max_steps"]:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)

        # ---- forward through student (with hidden states for distillation)
        out_s = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            token_type_ids=batch.get("token_type_ids"),
            labels=batch.get("labels"),
            output_hidden_states=True,
        )
        L_ft = out_s["loss"]
        loss = L_ft

        # ---- distillation (after teacher snapshot exists) ---------------
        if d["enable"] and step > p["pruning_start_step"]:
            sampled = sample_teacher_layers(n_layers, d["num_teacher_layers"], rng)
            t_logits, t_hidden = teacher_forward(model, batch, distiller, device)
            student_active = [True] * n_layers  # assume all layers retained
            phi = layer_mapping(student_active, sampled)
            distill = cofi_distill_loss(
                student_logits=out_s["logits"],
                teacher_logits=t_logits,
                student_hidden=list(out_s.get("hidden_states", [])),
                teacher_hidden=list(t_hidden),
                transforms=distiller.transforms,
                sampled_layers=sampled,
                phi=phi,
                temperature=distiller.temperature,
                pred_loss_weight=distiller.pred_loss_weight,
                layer_loss_weight=distiller.layer_loss_weight,
            )
            mu = mu_schedule(step, p["pruning_start_step"], p["pruning_end_step"])
            loss = mu * distill["L_distill"] + (1.0 - mu) * L_ft

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.trainable_parameters(), max_norm=o["grad_clip"]
        )
        optimizer.step()
        scheduler.step()

        # ---- adaptive pruning step --------------------------------------
        if p["enable"]:
            stats = pruner.step(
                model,
                step,
                head_dim=getattr(model.backbone.config, "hidden_size", 768) // 12,
                num_heads=getattr(model.backbone.config, "num_attention_heads", 12),
            )
            if step % cfg["runtime"]["log_every"] == 0:
                print(
                    f"[step {step}] loss={loss.item():.4f} "
                    f"sparsity_target={stats.get('sparsity_target', 0):.2f}"
                )

        # ---- adaptive tuning step (rank growth) -------------------------
        if t["enable"]:
            grew = ranker.grow(model, step)
            if grew:
                # Rebuild optimizer because parameter shapes changed.
                params_adapter = [
                    pp for pp in model.trainable_parameters() if pp.dim() > 1
                ]
                params_other = [
                    pp for pp in model.trainable_parameters() if pp.dim() <= 1
                ]
                optimizer = AdamW(
                    [
                        {"params": params_adapter, "lr": o["lr_adapter"]},
                        {"params": params_other, "lr": o["lr_classifier"]},
                    ],
                    betas=tuple(o["betas"]),
                    eps=o["eps"],
                    weight_decay=o["weight_decay"],
                )
                scheduler = linear_warmup_scheduler(
                    optimizer, o["warmup_steps"] - step, o["max_steps"] - step
                )
                print(f"[step {step}] grew ranks for {len(grew)} adapters")

        # ---- snapshot teacher every K steps -----------------------------
        if d["enable"] and step % max(50, p["pruning_start_step"] // 2) == 0:
            distiller.snapshot_teacher(model)

        # ---- periodic eval ----------------------------------------------
        if step > 0 and step % cfg["runtime"]["eval_every"] == 0:
            metrics = evaluate(model, eval_loader, device)
            metrics["step"] = step
            metrics_history.append(metrics)
            print(f"[step {step}] eval = {metrics}")

        step += 1

    # --- 5) Final evaluation + memory bookkeeping --------------------------
    final = evaluate(model, eval_loader, device)
    final["wall_time_sec"] = time.time() - t0
    if torch.cuda.is_available():
        # Addendum: report torch.cuda.max_memory_allocated()
        final["max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
    final["trainable_params"] = model.num_trainable()
    final["total_params"] = model.num_total()
    final["task"] = cfg["data"]["task_name"]
    final["primary_metric"] = get_task_metric(cfg["data"]["task_name"])

    print(f"[APT] FINAL = {final}")
    out_path = out_dir / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"final": final, "history": metrics_history}, f, indent=2)
    print(f"[APT] wrote {out_path}")


if __name__ == "__main__":
    main()
