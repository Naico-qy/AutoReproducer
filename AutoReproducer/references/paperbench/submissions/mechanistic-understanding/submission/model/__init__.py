"""Model components for the DPO toxicity case-study reproduction.

Exposes:
    GPT2WithHooks            -- GPT-2 wrapper that exposes residual streams,
                                 MLP key/value vectors and per-neuron activations.
    LinearToxicityProbe      -- the W_toxic probe used in Section 3.1.
    extract_toxic_value_vectors -- Section 3.1 cosine-similarity selection.
    svd_decompose_toxic_vectors  -- Section 3.1 SVD step (per addendum: SVD on d x N, not N x d).
    dpo_loss                 -- Equation in Section 4.1 (Rafailov et al. 2023).
"""

from .architecture import (
    GPT2WithHooks,
    LinearToxicityProbe,
    extract_toxic_value_vectors,
    svd_decompose_toxic_vectors,
    dpo_loss,
)

__all__ = [
    "GPT2WithHooks",
    "LinearToxicityProbe",
    "extract_toxic_value_vectors",
    "svd_decompose_toxic_vectors",
    "dpo_loss",
]
