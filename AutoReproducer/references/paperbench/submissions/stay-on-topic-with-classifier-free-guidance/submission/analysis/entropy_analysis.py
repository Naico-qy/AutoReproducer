"""§5.1 -- Effect of CFG on per-token sampling entropy.

Reproduces the comparison reported as:
    "CFG entropy distribution is significantly lower across generation
     steps than vanilla prompting, with a mean of 4.7 vs. 5.49"

The entropy formula is fixed by the addendum to:
    H(p) = -Σ_k p_k log p_k
computed via `scipy.stats.entropy`, averaged per-token over a generation,
then averaged over samples.

We report TWO summaries:
    1. mean per-token entropy (the primary number in §5.1)
    2. number of tokens needed to cover top-p=0.9 of the distribution
       (the "restricts the number of tokens in the top-p=90%" claim)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.p3_sampler import P3SamplerConfig, sample_p3
from model.architecture import CFGSampler
from model.entropy import sequence_entropy


# ---------------------------------------------------------------------------
# Per-step entropy collection
# ---------------------------------------------------------------------------
@torch.no_grad()
def _per_token_distributions(
    sampler: CFGSampler,
    prompt: str,
    max_new_tokens: int,
) -> torch.Tensor:
    """Return one (max_new_tokens, V) tensor of per-step probabilities under CFG."""
    device = next(sampler.model.parameters()).device
    enc = sampler.tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc.input_ids
    uncond = sampler._build_unconditional(input_ids)

    cond_ids = input_ids
    uncond_ids = uncond
    probs_list: List[torch.Tensor] = []

    for _ in range(max_new_tokens):
        cond_logits = sampler.model(cond_ids).logits[:, -1, :]
        uncond_logits = sampler.model(uncond_ids).logits[:, -1, :]
        cond_log = F.log_softmax(cond_logits, dim=-1)
        uncond_log = F.log_softmax(uncond_logits, dim=-1)
        from model.architecture import cfg_combine_logits

        combined = cfg_combine_logits(
            cond_logits=cond_log,
            uncond_logits=uncond_log,
            gamma=sampler.guidance_scale,
            renormalize=True,
        )
        probs = combined.exp()
        probs_list.append(probs[0].detach().cpu())
        next_tok = probs.argmax(-1, keepdim=True)
        cond_ids = torch.cat([cond_ids, next_tok], dim=-1)
        uncond_ids = torch.cat([uncond_ids, next_tok], dim=-1)
        if (
            sampler.tokenizer.eos_token_id is not None
            and next_tok.item() == sampler.tokenizer.eos_token_id
        ):
            break
    return torch.stack(probs_list, dim=0) if probs_list else torch.empty(0)


def compute_top_p_token_count(probs: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
    """For each row, the smallest k such that the top-k mass exceeds top_p."""
    sorted_p, _ = probs.sort(dim=-1, descending=True)
    cumulative = sorted_p.cumsum(dim=-1)
    counts = (cumulative < top_p).sum(dim=-1) + 1
    return counts


# ---------------------------------------------------------------------------
# Top-level driver: §5.1 comparison
# ---------------------------------------------------------------------------
@dataclass
class EntropyComparison:
    mean_entropy_baseline: float
    mean_entropy_cfg: float
    mean_topp_count_baseline: float
    mean_topp_count_cfg: float
    n_samples: int


@torch.no_grad()
def compare_entropy(
    base_sampler: CFGSampler,  # γ=1 (vanilla)
    cfg_sampler: CFGSampler,  # γ>1 (treatment)
    p3_cfg: P3SamplerConfig,
    max_new_tokens: int = 64,
    top_p: float = 0.9,
    max_samples: int = 1000,
) -> EntropyComparison:
    """Mean per-token entropy under vanilla vs. CFG, on a P3 sample."""
    H_base, H_cfg = [], []
    K_base, K_cfg = [], []
    n = 0
    for sample in tqdm(
        sample_p3(p3_cfg, tokenizer=base_sampler.tokenizer),
        total=max_samples,
        desc="entropy",
    ):
        if n >= max_samples:
            break
        prompt = sample["input"]

        probs_b = _per_token_distributions(base_sampler, prompt, max_new_tokens)
        probs_c = _per_token_distributions(cfg_sampler, prompt, max_new_tokens)
        if probs_b.numel() == 0 or probs_c.numel() == 0:
            continue

        H_b = sequence_entropy(probs_b)
        H_c = sequence_entropy(probs_c)
        H_base.append(float(H_b.mean()))
        H_cfg.append(float(H_c.mean()))

        K_b = compute_top_p_token_count(probs_b, top_p=top_p).float()
        K_c = compute_top_p_token_count(probs_c, top_p=top_p).float()
        K_base.append(float(K_b.mean()))
        K_cfg.append(float(K_c.mean()))
        n += 1

    return EntropyComparison(
        mean_entropy_baseline=float(np.mean(H_base)) if H_base else 0.0,
        mean_entropy_cfg=float(np.mean(H_cfg)) if H_cfg else 0.0,
        mean_topp_count_baseline=float(np.mean(K_base)) if K_base else 0.0,
        mean_topp_count_cfg=float(np.mean(K_cfg)) if K_cfg else 0.0,
        n_samples=n,
    )
