"""CFG-for-LLMs model package.

Public exports:
    CFGLogitsProcessor          -- HuggingFace LogitsProcessor that combines
                                   conditional / unconditional logits.
    CFGSampler                  -- Higher-level wrapper that runs both forward
                                   passes in lock-step.
    generate_with_cfg           -- One-shot helper for the eval scripts.
    NegativePromptCFG           -- Implements Eq. 5 (negative prompting).
    count_flops_electra         -- ELECTRA-style FLOP counter for §4.1.
    sequence_entropy            -- scipy.stats.entropy wrapper used in §5.1.
"""

from .architecture import (
    CFGLogitsProcessor,
    CFGSampler,
    generate_with_cfg,
    cfg_combine_logits,
)
from .negative_prompt import NegativePromptCFG
from .flops import count_flops_electra, ElectraFlopsConfig
from .entropy import sequence_entropy, mean_token_entropy

__all__ = [
    "CFGLogitsProcessor",
    "CFGSampler",
    "generate_with_cfg",
    "cfg_combine_logits",
    "NegativePromptCFG",
    "count_flops_electra",
    "ElectraFlopsConfig",
    "sequence_entropy",
    "mean_token_entropy",
]
