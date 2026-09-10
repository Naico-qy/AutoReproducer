"""ELECTRA-style FLOP counter -- §4.1.

The addendum specifies that FLOPs in §4.1 should be measured with the
ELECTRA flops_computation.py script:
    https://github.com/google-research/electra/blob/master/flops_computation.py

We re-implement the relevant parts of that script as a transparent helper
so that we can count FLOPs **per inference token** for any HuggingFace
config (GPT-2, Pythia, Falcon, CodeGen).

Conventions follow the original ELECTRA file:
    * 1 multiply + 1 add = 2 FLOPs (i.e. MAC = 2 FLOPs)
    * Softmax counted as ~5 FLOPs per element (exp + sub + div + sum)
    * Layer-norm counted as 5 FLOPs per element
    * GeLU counted as 8 FLOPs per element
    * Attention dot-products and softmax accounted for explicitly
"""

from __future__ import annotations

from dataclasses import dataclass


# Constants borrowed verbatim from electra/flops_computation.py
MAC = 2  # multiply-add = 2 ops
SOFTMAX_FLOPS = 5  # per element
LAYER_NORM_FLOPS = 5  # per element
ACTIVATION_FLOPS = 8  # GeLU per element
DROPOUT_FLOPS = 4  # per element (Bernoulli sample + multiply)


@dataclass
class ElectraFlopsConfig:
    """A direct mirror of the ELECTRA hparams that drive the formula.

    Pulled out so the same counter can serve every model in the §4.1 sweep.
    """

    h: int  # hidden size
    l: int  # num layers
    h_ff: int  # FFN intermediate size
    v: int  # vocab size
    s: int = 1024  # sequence length used for the sweep (LM-Eval default)
    e: int = 0  # tied embeddings: 0 if shared with output, else h
    is_decoder: bool = True  # we only handle decoders


# ---------------------------------------------------------------------------
# Block-by-block FLOPs (per token unless otherwise stated)
# ---------------------------------------------------------------------------
def embedding_flops(cfg: ElectraFlopsConfig) -> int:
    """Token + position embedding lookup is a sparse matmul ≈ negligible.

    The ELECTRA paper counts 1 lookup as h FLOPs per token.
    """
    return cfg.h


def attention_flops(cfg: ElectraFlopsConfig) -> int:
    """FLOPs of one attention block per token."""
    h, s = cfg.h, cfg.s
    # Q, K, V, O projections -- 4 matmuls of shape (h, h)
    qkv = 4 * MAC * h * h
    # QK^T: (h) × (s,h) -- per query token
    qk = MAC * h * s
    # softmax over s logits
    sm = SOFTMAX_FLOPS * s
    # PV: (s) × (s,h)
    pv = MAC * s * h
    return qkv + qk + sm + pv


def ffn_flops(cfg: ElectraFlopsConfig) -> int:
    """FLOPs of one feed-forward block per token."""
    h, h_ff = cfg.h, cfg.h_ff
    # Two linear projections + activation
    lin = 2 * MAC * h * h_ff
    act = ACTIVATION_FLOPS * h_ff
    return lin + act


def layer_norm_flops(cfg: ElectraFlopsConfig) -> int:
    return LAYER_NORM_FLOPS * cfg.h


def output_head_flops(cfg: ElectraFlopsConfig) -> int:
    """LM head projection h -> V plus softmax over V."""
    return MAC * cfg.h * cfg.v + SOFTMAX_FLOPS * cfg.v


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------
def count_flops_electra(cfg: ElectraFlopsConfig, with_cfg: bool = False) -> int:
    """Total inference FLOPs for one decoded token.

    Parameters
    ----------
    cfg       : ElectraFlopsConfig with model hyperparameters.
    with_cfg  : if True, doubles the cost (CFG runs two forward passes
                per token -- §4 "CFG requires two passes through the network,
                effectively doubling the amount of FLOPs required").
    """
    per_layer = attention_flops(cfg) + ffn_flops(cfg) + 2 * layer_norm_flops(cfg)
    body = cfg.l * per_layer
    total = embedding_flops(cfg) + body + output_head_flops(cfg)
    if with_cfg:
        # §4: CFG needs an additional forward pass per generation step.
        total *= 2
    return total


def hf_config_to_electra(hf_config) -> ElectraFlopsConfig:
    """Translate a HuggingFace `PretrainedConfig` into ElectraFlopsConfig.

    Robust to GPT-2 / Pythia / Falcon / CodeGen field-naming differences.
    """
    h = getattr(hf_config, "hidden_size", getattr(hf_config, "n_embd", None))
    l = getattr(hf_config, "num_hidden_layers", getattr(hf_config, "n_layer", None))
    h_ff = getattr(hf_config, "intermediate_size", getattr(hf_config, "n_inner", None))
    if h_ff is None and h is not None:
        h_ff = 4 * h  # GPT-2 default
    v = getattr(hf_config, "vocab_size", 50257)
    s = getattr(
        hf_config, "max_position_embeddings", getattr(hf_config, "n_positions", 1024)
    )
    return ElectraFlopsConfig(
        h=int(h or 0),
        l=int(l or 0),
        h_ff=int(h_ff or 0),
        v=int(v),
        s=int(s),
    )
