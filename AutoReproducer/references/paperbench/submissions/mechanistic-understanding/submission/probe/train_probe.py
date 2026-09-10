"""Train the W_toxic probe on Jigsaw (Section 3.1).

Per addendum.md, W_toxic is a [d_model, 2] matrix and the probe vector is
W_toxic[:, 1].  The probe takes as input the *mean-pooled last-layer residual
stream* of GPT-2 medium for a comment.

Reproduces:
    P(Toxic | x_bar^{L-1}) = softmax(W_toxic^T x_bar^{L-1})
    target validation accuracy: 0.94
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data import load_jigsaw
from model.architecture import GPT2WithHooks, LinearToxicityProbe


@torch.no_grad()
def extract_features(
    model: GPT2WithHooks, texts: list[str], batch_size: int = 32, max_len: int = 256
) -> torch.Tensor:
    """Mean-pool the last-layer residual stream across non-pad tokens."""
    device = next(model.parameters()).device
    feats: list[torch.Tensor] = []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size), desc="extract"):
        batch = texts[i : i + batch_size]
        enc = model.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.lm.transformer(**enc, output_hidden_states=False)
        hidden = out.last_hidden_state  # [B, T, d]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        feats.append(pooled.cpu())
    return torch.cat(feats, dim=0)  # [N, d]


def train_jigsaw_probe(
    model: GPT2WithHooks,
    save_path: str | Path,
    train_val_split: float = 0.9,
    batch_size: int = 32,
    epochs: int = 3,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    max_train: int | None = None,
) -> LinearToxicityProbe:
    """Train the linear probe and save W_toxic to ``save_path``.

    Returns the trained ``LinearToxicityProbe`` (so its ``toxic_direction``
    can be fed straight into ``extract_toxic_value_vectors``).
    """
    device = next(model.parameters()).device
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading Jigsaw...")
    raw = load_jigsaw(split="train", max_samples=max_train)

    # Section 3.1: 90:10 split
    n = len(raw)
    n_train = int(n * train_val_split)
    train_pairs = raw[:n_train]
    val_pairs = raw[n_train:]

    print(f"Extracting GPT-2 features for {n_train} train + {len(val_pairs)} val...")
    X_train = extract_features(
        model, [t for t, _ in train_pairs], batch_size=batch_size
    )
    y_train = torch.tensor([y for _, y in train_pairs], dtype=torch.long)
    X_val = extract_features(model, [t for t, _ in val_pairs], batch_size=batch_size)
    y_val = torch.tensor([y for _, y in val_pairs], dtype=torch.long)

    probe = LinearToxicityProbe(d_model=model.d_model).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    for ep in range(epochs):
        probe.train()
        running = 0.0
        for xb, yb in tqdm(train_loader, desc=f"probe ep{ep + 1}"):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = probe(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        print(f"  ep{ep + 1} train_loss={running / len(X_train):.4f}")

        probe.eval()
        with torch.no_grad():
            logits = probe(X_val.to(device))
            acc = (logits.argmax(-1) == y_val.to(device)).float().mean().item()
        print(f"  ep{ep + 1} val_acc={acc:.4f}  (paper target: 0.94)")

    # Save W_toxic = [d_model, 2]
    torch.save(
        {
            "W_toxic": probe.W_toxic.detach().cpu(),
            "toxic_direction": probe.toxic_direction.detach().cpu(),
            "state_dict": probe.state_dict(),
            "val_accuracy": acc,
        },
        save_path,
    )
    print(f"Saved probe -> {save_path}")
    return probe


if __name__ == "__main__":
    import argparse, yaml

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--max-train", type=int, default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2WithHooks(cfg["model"]["name"]).to(device)
    train_jigsaw_probe(
        model,
        save_path=cfg["output"]["probe_ckpt"],
        train_val_split=cfg["probe"]["train_val_split"],
        batch_size=cfg["probe"]["batch_size"],
        epochs=cfg["probe"]["epochs"],
        lr=cfg["probe"]["lr"],
        weight_decay=cfg["probe"]["weight_decay"],
        max_train=args.max_train,
    )
