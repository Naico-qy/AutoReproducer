"""K-step model refinement on a single online example (x_i, y_i).

Per §4.1 (Hyperparameters):
    * 30 steps of parameter updates for LoRA / Full-FT.
    * 100 steps when only fine-tuning LM heads.
    * lr = 1e-5 (BART0 single error), 1e-4 (FLAN-T5 single error).
    * Smaller lrs (1e-6 / 1e-5) for sequential refinement.
    * lr = 1e-3 / 1e-4 for head-only refinement.
"""

from __future__ import annotations

from typing import Optional

try:
    import torch
    from torch.optim import AdamW
except Exception:  # pragma: no cover - keep importable on CPU-only CI
    torch = None  # type: ignore
    AdamW = None  # type: ignore


def _seq2seq_loss(model, tokenizer, x: str, y: str, device):
    """Standard teacher-forced cross-entropy loss for an encoder-decoder LM."""
    enc = tokenizer(
        x,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)
    with (
        tokenizer.as_target_tokenizer()
        if hasattr(tokenizer, "as_target_tokenizer")
        else _nullctx()
    ):
        labels = tokenizer(
            y,
            return_tensors="pt",
            truncation=True,
            max_length=64,
        ).input_ids.to(device)
    labels[labels == tokenizer.pad_token_id] = -100
    out = model(
        input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=labels
    )
    return out.loss


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def refine_one_step(model, tokenizer, x: str, y: str, optimizer, device) -> float:
    """Perform a single optimization step on (x, y); return loss value."""
    optimizer.zero_grad()
    loss = _seq2seq_loss(model, tokenizer, x, y, device)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def refine_K_steps(
    model,
    tokenizer,
    x: str,
    y: str,
    K: int,
    lr: float,
    device,
    weight_decay: float = 0.0,
    grad_clip: Optional[float] = 1.0,
    replay_step_fn=None,
    replay_every: int = 0,
) -> list[float]:
    """Run K SGD/AdamW steps on (x_i, y_i), interleaving replay if requested.

    `replay_step_fn(model, optimizer, device, t)` is called every
    `replay_every` steps for replay-based refinement (§4.2).
    """
    if torch is None:
        raise RuntimeError("PyTorch not available; install requirements.txt")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    losses: list[float] = []
    for t in range(K):
        optimizer.zero_grad()
        loss = _seq2seq_loss(model, tokenizer, x, y, device)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))

        # interleaved replay (§4.2): batch_size every `replay_every` steps
        if (
            replay_step_fn is not None
            and replay_every > 0
            and (t + 1) % replay_every == 0
        ):
            replay_step_fn(model, optimizer, device, t)

    return losses
