"""Negative-prompting wrapper -- Equation 5 of the paper.

Equation 5 generalises CFG by replacing the unconditional context with an
explicit *negative* prompt c̄:

    log P_hat(w_i | w_{<i}, c, c̄) = log P(w_i | w_{<i}, c̄)
                                 + γ · (log P(w_i | w_{<i}, c)
                                      - log P(w_i | w_{<i}, c̄))            (Eq. 5)

When c̄ = ∅ (the empty / dropped prefix) Eq. 5 reduces to Eq. 7.

The §3.4 chatbot experiments use:
    c   = edited system prompt   ("...write a sad response.")
    c̄  = default system prompt   ("...write an appropriate response.")

§3.4 itself is **out of scope** for reproduction (human study), but the
*mechanism* of negative prompting is in scope and is reused in §5.3 for
the visualisation experiment (Table 3, c̄ = ∅).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .architecture import cfg_combine_logits, CFGLogitsProcessor


@dataclass
class NegativePromptCFG:
    """Convenience wrapper that performs Eq. 5 sampling.

    Compared with the bare ``CFGSampler`` (which assumes c̄ = ∅) this class
    accepts BOTH c (positive prompt) AND c̄ (negative prompt) as text inputs.
    """

    model: object
    tokenizer: object
    guidance_scale: float = 3.0  # §3.4 reports peak at γ=3
    renormalize: bool = True

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        positive_prompt: str,
        negative_prompt: str,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **gen_kwargs,
    ) -> str:
        device = next(self.model.parameters()).device
        pos = self.tokenizer(positive_prompt, return_tensors="pt").to(device)
        neg = self.tokenizer(negative_prompt, return_tensors="pt").to(device)

        processor = CFGLogitsProcessor(
            guidance_scale=self.guidance_scale,
            model=self.model,
            unconditional_ids=neg.input_ids,
            unconditional_attention_mask=neg.attention_mask,
            renormalize=self.renormalize,
        )

        from transformers import LogitsProcessorList

        logits_processor = LogitsProcessorList([processor])
        out = self.model.generate(
            input_ids=pos.input_ids,
            attention_mask=pos.attention_mask,
            logits_processor=logits_processor,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )
        return self.tokenizer.decode(
            out[0][pos.input_ids.shape[-1] :], skip_special_tokens=True
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step_logits(
        self,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Single-step Eq. 5 logits.  Used by the §5.3 visualizer."""
        cond = self.model(input_ids=positive_ids).logits[:, -1, :]
        uncond = self.model(input_ids=negative_ids).logits[:, -1, :]
        cond_log = F.log_softmax(cond, dim=-1)
        uncond_log = F.log_softmax(uncond, dim=-1)
        return cfg_combine_logits(
            cond_logits=cond_log,
            uncond_logits=uncond_log,
            gamma=self.guidance_scale,
            renormalize=self.renormalize,
        )


def build_default_negative_prompt() -> str:
    """The §3.4 default system prompt c̄."""
    return (
        "The prompt below is a question to answer, a task to complete, or a "
        "conversation to respond to; decide which and write an appropriate "
        "response."
    )
