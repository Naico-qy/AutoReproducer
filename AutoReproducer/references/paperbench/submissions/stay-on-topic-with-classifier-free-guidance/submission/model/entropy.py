"""Entropy / perplexity helpers for §5.1 and §5.2.

The addendum is explicit:

    * Entropy is H(p) = -Σ_k p_k log p_k computed via
      `scipy.stats.entropy` (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html).
    * The "mean entropy" reported in §5.1 is averaged **per-token over a
      generation**, then averaged over samples:
          (1/n) Σ_{i=1..n} H(p(x_i | x_{<i}))
    * §5.2 perplexity is **continuation-only** -- the prompt log-likelihood
      is excluded.

This module wraps those definitions so the analysis scripts always agree
with the paper's formulae.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.stats import entropy as scipy_entropy
except ImportError:  # pragma: no cover
    scipy_entropy = None


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------
def sequence_entropy(probs: torch.Tensor, base: float = np.e) -> torch.Tensor:
    """Per-row entropy of a (..., V) probability tensor.

    We delegate to `scipy.stats.entropy` to match the addendum exactly.
    The function is vectorised: `probs` may have any leading shape.
    """
    if scipy_entropy is None:  # pragma: no cover
        # Fall back to a torch implementation that produces the same numbers.
        eps = 1e-12
        return -(probs * (probs.clamp_min(eps)).log() / np.log(base)).sum(dim=-1)

    p_np = probs.detach().cpu().numpy()
    flat = p_np.reshape(-1, p_np.shape[-1])
    H = np.array([scipy_entropy(p, base=base) for p in flat])
    return torch.from_numpy(H.reshape(p_np.shape[:-1])).to(probs.device)


def mean_token_entropy(per_token_H: Iterable[torch.Tensor]) -> float:
    """Compute (1/n) Σ H(p_i) for one sequence."""
    if isinstance(per_token_H, torch.Tensor):
        return float(per_token_H.float().mean().item())
    arr = np.array([h.item() if hasattr(h, "item") else float(h) for h in per_token_H])
    return float(arr.mean()) if arr.size else 0.0


# ---------------------------------------------------------------------------
# Perplexity helpers (§5.2)
# ---------------------------------------------------------------------------
def continuation_perplexity(
    model,
    tokenizer,
    prompt: str,
    continuation: str,
) -> float:
    """Perplexity of `continuation` given `prompt`, ignoring prompt loss.

    Returns exp(NLL / N_continuation_tokens), matching the addendum's
    definition for §5.2.
    """
    device = next(model.parameters()).device
    full_ids = tokenizer(prompt + continuation, return_tensors="pt").input_ids.to(
        device
    )
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    n_prompt = prompt_ids.shape[-1]
    n_total = full_ids.shape[-1]
    if n_total <= n_prompt:
        return float("nan")

    with torch.no_grad():
        logits = model(input_ids=full_ids).logits  # (1, T, V)
    # Shift for next-token loss
    shift_logits = logits[:, :-1, :]
    shift_labels = full_ids[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Mask out prompt tokens -- §5.2 ignores prompt loss (addendum)
    mask = torch.zeros_like(token_logp)
    mask[:, n_prompt - 1 :] = 1.0
    nll = -(token_logp * mask).sum() / mask.sum().clamp_min(1)
    return float(torch.exp(nll).item())


def cfg_continuation_perplexity(
    sampler,  # CFGSampler
    prompt: str,
    continuation: str,
) -> float:
    """Continuation-only PPL under CFG (§5.2)."""
    logp = sampler.loglikelihood(prompt, continuation)
    n = len(sampler.tokenizer(continuation).input_ids)
    if n == 0:
        return float("nan")
    return float(np.exp(-logp / n))
