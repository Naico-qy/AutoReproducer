"""Optional training entrypoint for "Stay on topic with CFG".

The paper makes the strong claim (§1, §2.2, Conclusion) that CFG works
**out-of-the-box** in autoregressive language models -- no extra training
is needed.  Quoting §2.2:

    "language models handle both P_θ(w|c) and P_θ(w) naturally due to being
     trained on finite context windows.  In other words, dropping the prefix
     c is a natural feature."

Training is therefore *not* part of the reproduction.  However, for
completeness, this script implements the **conditioning-dropout** style
fine-tuning that is required for CFG in the diffusion setting (Ho &
Salimans, 2021), so that it is available for any reader who wishes to
apply CFG in a trained-from-scratch / fine-tuned regime.

The procedure mirrors the dropout-on-condition recipe used in §6 of
the original CFG paper:

    With probability p_drop, drop the prompt prefix and train the LM to
    predict the continuation unconditionally; otherwise train conditionally
    as usual.  This makes the same network model both P_θ(w | c) and
    P_θ(w), which is the prerequisite for CFG sampling.

This is invoked via:

    python train.py --config configs/default.yaml --output_dir runs/cfg_ft
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherits", None)
    if parent is not None:
        parent_path = Path(path).parent / parent
        base = _load_yaml(str(parent_path))
        merged = {**base, **cfg}
        for k in ("cfg", "generation"):
            if k in base and k in cfg and isinstance(base[k], dict):
                merged[k] = {**base[k], **cfg[k]}
        return merged
    return cfg


def _conditioning_dropout_collator(
    p_drop: float, tokenizer, prompt_key="prompt", target_key="completion"
):
    """Build a collator that randomly drops the prompt prefix.

    p_drop = probability that the prompt is replaced with the empty / BOS
    sequence so the LM learns the unconditional distribution P_θ(w).
    """
    import random

    def collate(batch):
        import torch

        input_ids = []
        labels = []
        max_len = 0
        for example in batch:
            prompt = example.get(prompt_key, "")
            target = example.get(target_key, "")
            if random.random() < p_drop:
                # Unconditional pass -- replace the prompt with a single BOS.
                bos = tokenizer.bos_token_id or tokenizer.eos_token_id
                pids = [bos]
            else:
                pids = tokenizer(prompt, add_special_tokens=False).input_ids
            tids = tokenizer(target, add_special_tokens=False).input_ids
            ids = pids + tids
            lbl = [-100] * len(pids) + tids
            input_ids.append(ids)
            labels.append(lbl)
            max_len = max(max_len, len(ids))

        pad = tokenizer.pad_token_id or 0
        for i in range(len(input_ids)):
            n = max_len - len(input_ids[i])
            input_ids[i] = input_ids[i] + [pad] * n
            labels[i] = labels[i] + [-100] * n

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--output_dir", default="runs/cfg_ft")
    p.add_argument("--model_id", default="EleutherAI/pythia-410m")
    p.add_argument("--dataset", default="bigscience/P3")
    p.add_argument(
        "--p_drop",
        type=float,
        default=0.1,
        help="Conditioning-dropout rate (Ho & Salimans 2021).",
    )
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=10_000)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Lazy imports so users on a CPU-only box can still inspect the script.
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("`datasets` is required for training.") from e

    cfg = _load_yaml(args.config)
    print("Loaded config:", cfg.get("seed"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    ds = load_dataset(args.dataset, split="train", streaming=True)
    collate = _conditioning_dropout_collator(
        args.p_drop,
        tokenizer,
        prompt_key="inputs_pretokenized",
        target_key="targets_pretokenized",
    )
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_steps)

    step = 0
    for _epoch in range(args.epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"step {step:6d}  loss {loss.item():.4f}")
            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved CFG-finetuned model to {args.output_dir}")


if __name__ == "__main__":
    main()
