"""Core Classifier-Free Guidance for autoregressive language models.

Implements Equations 6 and 7 from
"Stay on topic with Classifier-Free Guidance" (Sanchez et al., ICML 2024):

    log P_hat(w_i | w_{<i}, c) = log P(w_i | w_{<i})
                                + γ · (log P(w_i | w_{<i}, c)
                                     - log P(w_i | w_{<i}))                  (Eq. 7)

Design notes
------------
* The combination is performed in **logits space**, not in noise/PDF space.
  The paper makes this explicit (§2.2: "Here, we apply CFG to the logits of
  next-token predictions").
* The unconditional pass uses the input where the prompt prefix `c` is
  dropped.  The paper (§3.1) uses "the last token of the initial prompt" as
  the unconditional context — a single-token decoder state on which the LM
  produces a perfectly natural distribution because LMs are trained on
  finite-context windows (§2.2: "dropping the prefix c is a natural feature").
* Two forward passes per decoding step doubles inference FLOPs (§4 — see
  `flops.py`).
* Equation 7 is mathematically identical to a linear combination of logits
  followed by re-normalisation; we implement it directly without exponential
  re-scaling.

This module is fully compatible with HuggingFace `transformers.generate`
through the `LogitsProcessor` interface, mirroring the upstream
`UnbatchedClassifierFreeGuidanceLogitsProcessor` that the authors contributed
to the library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

try:
    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizerBase,
        LogitsProcessor,
        LogitsProcessorList,
    )
except ImportError:  # pragma: no cover -- syntax-only environments
    PreTrainedModel = object  # type: ignore
    PreTrainedTokenizerBase = object  # type: ignore

    class LogitsProcessor:  # type: ignore
        def __call__(self, input_ids, scores):
            raise NotImplementedError

    class LogitsProcessorList(list):  # type: ignore
        pass


# ---------------------------------------------------------------------------
# Pure functional CFG combine -- safe to call from anywhere
# ---------------------------------------------------------------------------
def cfg_combine_logits(
    cond_logits: torch.Tensor,
    uncond_logits: torch.Tensor,
    gamma: float,
    renormalize: bool = True,
) -> torch.Tensor:
    """Apply Eq. 7 in log-space.

    Parameters
    ----------
    cond_logits   : Tensor [..., V]   log P_θ(w_i | w_{<i}, c)
    uncond_logits : Tensor [..., V]   log P_θ(w_i | w_{<i})
    gamma         : float             guidance weight γ.  γ=1 reduces to the
                                      conditional model; γ=0 to the
                                      unconditional model.
    renormalize   : bool              whether to subtract logsumexp so that
                                      the output is a normalised log-prob
                                      distribution (numerically safer).
    """
    # We work in log-space throughout (the inputs come from log_softmax).
    combined = uncond_logits + gamma * (cond_logits - uncond_logits)
    if renormalize:
        # log-softmax to get a normalised distribution
        combined = combined - torch.logsumexp(combined, dim=-1, keepdim=True)
    return combined


# ---------------------------------------------------------------------------
# HuggingFace LogitsProcessor implementation
# ---------------------------------------------------------------------------
class CFGLogitsProcessor(LogitsProcessor):
    """`generate(...)`-compatible logits processor.

    On each decoding step the processor:
        1.  Re-runs the model with the unconditional input (prompt-dropped).
        2.  Combines the conditional `scores` with the unconditional logits
            via Eq. 7.

    The unconditional context is built once at construction time and grown
    auto-regressively as decoding advances.  We use `past_key_values` to
    avoid re-encoding the unconditional prefix.

    This mirrors `transformers.UnbatchedClassifierFreeGuidanceLogitsProcessor`
    (HF docs reference: ``UnbatchedCFG``); we re-implement it here for
    transparency and because the paper's formulation is the canonical one.
    """

    def __init__(
        self,
        guidance_scale: float,
        model: PreTrainedModel,
        unconditional_ids: torch.Tensor,
        unconditional_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        renormalize: bool = True,
    ) -> None:
        self.guidance_scale = float(guidance_scale)
        self.model = model
        self.unconditional_ids = unconditional_ids
        self.unconditional_attention_mask = unconditional_attention_mask
        self.use_cache = use_cache
        self.renormalize = renormalize
        self._uncond_past: Optional[tuple] = None
        self._uncond_input_len: int = unconditional_ids.shape[-1]

    # ------------------------------------------------------------------
    def _uncond_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the unconditional pass, growing it by the newly sampled tokens.

        We feed exactly one new token per step using `past_key_values`.
        """
        if self._uncond_past is None:
            # First call -- prime the model with the unconditional prefix.
            outputs = self.model(
                input_ids=self.unconditional_ids,
                attention_mask=self.unconditional_attention_mask,
                use_cache=self.use_cache,
                return_dict=True,
            )
            self._uncond_past = outputs.past_key_values
            return outputs.logits[:, -1, :]

        # Extract just the most recent token from `input_ids` (which contains
        # the entire conditional sequence so far).
        new_token = input_ids[:, -1:]
        outputs = self.model(
            input_ids=new_token,
            past_key_values=self._uncond_past,
            use_cache=self.use_cache,
            return_dict=True,
        )
        self._uncond_past = outputs.past_key_values
        return outputs.logits[:, -1, :]

    # ------------------------------------------------------------------
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # γ = 1 => no-op (Eq. 7 reduces to the conditional logits).
        if self.guidance_scale == 1.0:
            return scores

        # `scores` is the conditional log-prob distribution at this step.
        cond_log = F.log_softmax(scores, dim=-1)
        uncond_logits = self._uncond_forward(input_ids)
        uncond_log = F.log_softmax(uncond_logits, dim=-1)
        return cfg_combine_logits(
            cond_logits=cond_log,
            uncond_logits=uncond_log,
            gamma=self.guidance_scale,
            renormalize=self.renormalize,
        )


# ---------------------------------------------------------------------------
# High-level helper
# ---------------------------------------------------------------------------
@dataclass
class CFGSampler:
    """High-level wrapper that produces sequences with CFG.

    Parameters
    ----------
    model              : a CausalLM `PreTrainedModel`.
    tokenizer          : matching tokenizer.
    guidance_scale     : γ in Eq. 7.
    uncond_from_last_token : bool — if True, the unconditional context is
        the *last token* of the prompt (Sec. 3.1's recipe).  If False, the
        unconditional context is the empty string / BOS token.
    negative_prompt    : optional `c̄` for Eq. 5 (overrides the
        last-token / empty unconditional prefix).
    """

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    guidance_scale: float = 1.5
    uncond_from_last_token: bool = True
    negative_prompt: Optional[str] = None
    renormalize: bool = True

    # ------------------------------------------------------------------
    def _build_unconditional(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Build the unconditional input ids per the paper's recipe."""
        if self.negative_prompt is not None:
            # Eq. 5: c̄ is given explicitly
            return self.tokenizer(
                self.negative_prompt,
                return_tensors="pt",
            ).input_ids.to(prompt_ids.device)

        if self.uncond_from_last_token:
            # §3.1: last token of the prompt as the unconditional context
            return prompt_ids[:, -1:].clone()

        # Empty / BOS fallback
        bos = self.tokenizer.bos_token_id
        if bos is None:
            bos = self.tokenizer.eos_token_id
        return torch.tensor([[bos]], device=prompt_ids.device)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        **gen_kwargs,
    ) -> str:
        """One-shot generation with CFG."""
        device = next(self.model.parameters()).device
        enc = self.tokenizer(prompt, return_tensors="pt").to(device)
        uncond_ids = self._build_unconditional(enc.input_ids)

        processor = CFGLogitsProcessor(
            guidance_scale=self.guidance_scale,
            model=self.model,
            unconditional_ids=uncond_ids,
            renormalize=self.renormalize,
        )
        logits_processor = LogitsProcessorList([processor])

        out = self.model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            logits_processor=logits_processor,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )
        return self.tokenizer.decode(
            out[0][enc.input_ids.shape[-1] :], skip_special_tokens=True
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def loglikelihood(self, context: str, continuation: str) -> float:
        """Compute log P_hat(continuation | context) under CFG.

        Used by the LM-Eval harness style zero-shot scoring (§3.1).
        Continuation-only loglikelihood -- prompt loss is ignored
        (the addendum makes this explicit for §5.2).
        """
        device = next(self.model.parameters()).device
        ctx_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(device)
        full_ids = self.tokenizer(
            context + continuation, return_tensors="pt"
        ).input_ids.to(device)
        cont_ids = full_ids[:, ctx_ids.shape[-1] :]
        if cont_ids.numel() == 0:
            return 0.0

        # Conditional pass over the full sequence
        cond_logits = self.model(input_ids=full_ids).logits  # (1, T, V)
        # Shifted log-probabilities for the continuation
        cond_log = F.log_softmax(cond_logits[:, ctx_ids.shape[-1] - 1 : -1, :], dim=-1)

        # Unconditional pass: prefix is the unconditional context, then we
        # extend it with the continuation tokens.
        uncond_prefix = self._build_unconditional(ctx_ids)
        uncond_seq = torch.cat([uncond_prefix, cont_ids], dim=-1)
        uncond_logits = self.model(input_ids=uncond_seq).logits
        uncond_log = F.log_softmax(
            uncond_logits[:, uncond_prefix.shape[-1] - 1 : -1, :], dim=-1
        )

        combined = cfg_combine_logits(
            cond_logits=cond_log,
            uncond_logits=uncond_log,
            gamma=self.guidance_scale,
            renormalize=self.renormalize,
        )
        # Gather per-token continuation log-probs and sum
        token_logp = combined.gather(-1, cont_ids.unsqueeze(-1)).squeeze(-1)
        return float(token_logp.sum().item())


# ---------------------------------------------------------------------------
# Convenience for the eval scripts
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_with_cfg(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    guidance_scale: float = 1.5,
    negative_prompt: Optional[str] = None,
    uncond_from_last_token: bool = True,
    **gen_kwargs,
) -> str:
    """One-call CFG generation -- thin wrapper around CFGSampler."""
    sampler = CFGSampler(
        model=model,
        tokenizer=tokenizer,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        uncond_from_last_token=uncond_from_last_token,
    )
    return sampler.generate(prompt, **gen_kwargs)
