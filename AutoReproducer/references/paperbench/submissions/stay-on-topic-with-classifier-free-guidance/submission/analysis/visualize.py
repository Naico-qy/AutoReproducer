"""§5.3 -- Token re-ranking visualization (Table 3).

Given a positive prompt c (and optionally c̄), at each generation step we
rank the vocabulary by the difference

    Δ(w_t) = log P(w_t | w_{<t}, c) - log P(w_t | w_{<t})

This shows which tokens CFG most encourages and most discourages.

The paper's Table 3 example uses:
    c  = "The dragon flew over Paris, France"
    c̄ = ∅
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from model.architecture import CFGSampler


@dataclass
class StepRanking:
    step: int
    current_token: str
    top: List[str]
    bottom: List[str]


@torch.no_grad()
def token_reranking_table(
    sampler: CFGSampler,
    prompt: str,
    num_steps: int = 28,
    top_k: int = 5,
    bottom_k: int = 5,
    negative_prompt: Optional[str] = None,
) -> List[StepRanking]:
    """Build the per-step token-reranking table used in §5.3 / Table 3."""
    tokenizer = sampler.tokenizer
    model = sampler.model
    device = next(model.parameters()).device

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    if negative_prompt:
        neg_ids = tokenizer(negative_prompt, return_tensors="pt").input_ids.to(device)
    else:
        neg_ids = enc.input_ids[:, -1:].clone()  # last-token unconditional

    cond_ids = enc.input_ids
    uncond_ids = neg_ids
    out: List[StepRanking] = []

    for t in range(num_steps):
        cond_logits = model(cond_ids).logits[:, -1, :]
        uncond_logits = model(uncond_ids).logits[:, -1, :]
        cond_log = F.log_softmax(cond_logits, dim=-1)
        uncond_log = F.log_softmax(uncond_logits, dim=-1)
        delta = (cond_log - uncond_log)[0]  # (V,)

        topv, topi = torch.topk(delta, top_k)
        botv, boti = torch.topk(-delta, bottom_k)
        top_tokens = [tokenizer.decode([int(i)]).strip() for i in topi.tolist()]
        bot_tokens = [tokenizer.decode([int(i)]).strip() for i in boti.tolist()]
        cur_tok = tokenizer.decode([int(cond_ids[0, -1])]).strip()
        out.append(
            StepRanking(
                step=t, current_token=cur_tok, top=top_tokens, bottom=bot_tokens
            )
        )

        # Greedy decoding of the next token under CFG to advance the context
        from model.architecture import cfg_combine_logits

        comb = cfg_combine_logits(
            cond_log, uncond_log, gamma=sampler.guidance_scale, renormalize=True
        )
        next_tok = comb.argmax(-1, keepdim=True)
        cond_ids = torch.cat([cond_ids, next_tok], dim=-1)
        uncond_ids = torch.cat([uncond_ids, next_tok], dim=-1)
        if (
            tokenizer.eos_token_id is not None
            and int(next_tok.item()) == tokenizer.eos_token_id
        ):
            break

    return out
