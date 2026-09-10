"""Replay with distillation loss against the base PTLM f_0 (§4.2).

Per §4.2 of the paper:
    "We perform replay with a distillation loss against the outputs of
     the base PTLM (Buzzega et al., 2020a)."

That reference is *Dark Experience Replay* (DER), where the replay loss
is a soft-target distillation MSE/KL between the cached logits of f_0 on
each upstream example and the current model's logits.

Verified citation (closest baseline cited in §4.2):
    Aljundi et al. "Online Continual Learning with Maximally Interfered
    Retrieval." NeurIPS 2019. arXiv:1908.04742.
    (Verified via paper_search; CrossRef has no DOI for the NeurIPS pre-print.)
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None


# ---------------------------------------------------------------------
@dataclass
class ReplayItem:
    x: str
    y: str
    cached_logits: object  # tensor or None; from f_0 (Dark ER)


class ReplayBuffer:
    """Stores upstream examples and their cached f_0 logits."""

    def __init__(self, capacity: int = 100_000, seed: int = 0):
        self.capacity = capacity
        self.items: list[ReplayItem] = []
        self.rng = random.Random(seed)

    def add(self, x: str, y: str, cached_logits=None) -> None:
        if len(self.items) >= self.capacity:
            self.items.pop(0)
        self.items.append(ReplayItem(x=x, y=y, cached_logits=cached_logits))

    def __len__(self) -> int:
        return len(self.items)

    def sample(self, k: int) -> list[ReplayItem]:
        if not self.items:
            return []
        return self.rng.sample(self.items, k=min(k, len(self.items)))


# ---------------------------------------------------------------------
def distillation_replay_step(
    model,
    tokenizer,
    buffer: ReplayBuffer,
    optimizer,
    device,
    batch_size: int = 8,
    kd_alpha: float = 0.5,
    kd_temperature: float = 1.0,
) -> float:
    """One replay step (§4.2, Buzzega et al. 2020a / Dark ER).

    Loss = (1 - α) * cross-entropy on (x, y) ground-truth tokens
         +    α    * KL( current_logits / T  ||  f_0_logits / T ) * T^2

    Falls back to plain CE if no cached logits are available.
    """
    if torch is None:
        raise RuntimeError("PyTorch not available")

    items = buffer.sample(batch_size)
    if not items:
        return 0.0

    losses = []
    for it in items:
        enc = tokenizer(it.x, return_tensors="pt", truncation=True, max_length=512).to(
            device
        )
        labels = tokenizer(
            it.y, return_tensors="pt", truncation=True, max_length=64
        ).input_ids.to(device)
        labels_for_loss = labels.clone()
        labels_for_loss[labels_for_loss == tokenizer.pad_token_id] = -100

        out = model(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            labels=labels_for_loss,
            decoder_input_ids=None,
        )
        ce = out.loss
        kd = torch.tensor(0.0, device=device)
        if it.cached_logits is not None and isinstance(it.cached_logits, torch.Tensor):
            cur = out.logits  # (B, T, V)
            T_ = min(cur.size(1), it.cached_logits.size(0))
            V_ = min(cur.size(2), it.cached_logits.size(-1))
            cur_ = cur[0, :T_, :V_] / kd_temperature
            tgt_ = it.cached_logits[:T_, :V_].to(device) / kd_temperature
            kd = F.kl_div(
                F.log_softmax(cur_, dim=-1),
                F.softmax(tgt_, dim=-1),
                reduction="batchmean",
            ) * (kd_temperature**2)
        losses.append((1 - kd_alpha) * ce + kd_alpha * kd)

    optimizer.zero_grad()
    loss = torch.stack(losses).mean()
    loss.backward()
    optimizer.step()
    return float(loss.item())
