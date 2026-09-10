"""Toxicity-vector interventions (Section 3.3) and un-aligning (Section 6).

Implements four operations used by the paper:

  apply_residual_subtraction
        Section 3.3:  x^{L-1} <- x^{L-1} - alpha * W
        where W is one of {W_toxic[:,1], MLP.v_770^19, SVD.U_toxic[0], ...}.

  apply_un_align_key_scaling
        Section 6:    k_i^l <- 10 * k_i^l   for the top-N MLP.k_toxic.
        Reverts GPT2_DPO back to its toxic behaviour by enlarging the
        activation regions gamma(MLP.k_toxic^l).

  measure_logit_lens
        Figure 1.  Apply the unembedding layer to every intermittent
        residual stream and return p(token | x^l) at each layer.

  measure_mean_activations
        Figure 2 / 5.  Mean of m_i^l = sigma(W_K^l x^l) over a batch of
        prompts, for a given list of (layer, idx) MLP value vectors.

  compute_residual_offset
        Section 5.2.  delta_x^{l_mid} := x_DPO^{l_mid} - x_GPT2^{l_mid}.
        The residual-stream shift between the pre- and post-DPO models
        for the same input.
"""

from __future__ import annotations

import torch

from model.architecture import GPT2WithHooks


# ---------------------------------------------------------------------------
# Section 3.3 -- intervene by subtracting a toxic vector
# ---------------------------------------------------------------------------
def apply_residual_subtraction(
    model: GPT2WithHooks,
    vector: torch.Tensor,
    alpha: float,
    layer: int | None = None,
):
    """Subtract ``alpha * vector`` from the residual stream at ``layer``.

    Default layer is the last layer (layer = n_layers, i.e. after the final
    transformer block, just before the unembedding).  Equation Section 3.3.
    """
    if layer is None:
        layer = model.n_layers
    model.install_residual_subtraction(layer, vector, alpha)


# ---------------------------------------------------------------------------
# Section 6 -- un-align by scaling toxic key vectors
# ---------------------------------------------------------------------------
def apply_un_align_key_scaling(
    model: GPT2WithHooks,
    toxic_indices: list[tuple[int, int]],
    scale: float = 10.0,
    n_top: int = 7,
):
    """Section 6, Table 4: scale the top-7 MLP.k_toxic by 10x.

    Implements the "SCALE MLP.k_Toxic" row of Table 4.
    Selecting only the top-N keeps the perturbation localised, so PPL stays
    near the post-DPO baseline (~23.30) while toxicity reverts to ~0.46.
    """
    selected = toxic_indices[:n_top]
    scaling = {(layer, idx): scale for (layer, idx) in selected}
    model.install_key_scaling(scaling)


# ---------------------------------------------------------------------------
# Figure 1 -- logit lens
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_logit_lens(
    model: GPT2WithHooks,
    input_ids: torch.Tensor,
    target_token_id: int,
) -> dict:
    """Apply the unembedding layer to every intermittent residual stream.

    Returns a dict ``{layer: avg_prob_of_target}`` for both ``x^l`` and
    ``x^{l-mid}``.  Reproduces Figure 1 (probability of "sh*t" through layers).
    """
    _, cache = model.forward_with_cache(input_ids)
    ln_f = model.lm.transformer.ln_f  # final layer norm
    U = model.lm.lm_head.weight  # [|V|, d_model]

    out = {}
    for layer in range(model.n_layers):
        for tag, src in (("x", cache.x), ("x_mid", cache.x_mid)):
            if layer not in src:
                continue
            h = src[layer]  # [B, T, d]
            h = ln_f(h)  # logit-lens convention
            logits = h @ U.T  # [B, T, |V|]
            # average over batch and the *last* token (the prediction position)
            p = torch.softmax(logits[:, -1, :], dim=-1)[:, target_token_id]
            out[f"{tag}_{layer}"] = float(p.mean().item())
    return out


# ---------------------------------------------------------------------------
# Figure 2 / 5 -- mean activations m_i^l
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_mean_activations(
    model: GPT2WithHooks,
    input_ids_batch: list[torch.Tensor],
    selected: list[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    """Mean of m_i^l = sigma(W_K^l x^l) over the supplied prompts.

    For each (layer, idx) in ``selected`` we average across batch *and*
    sequence (last 20 tokens, matching the addendum: "the 20 tokens used to
    measure mean activation were greedily sampled from GPT2").
    """
    sums = {key: 0.0 for key in selected}
    counts = {key: 0 for key in selected}

    for ids in input_ids_batch:
        _, cache = model.forward_with_cache(ids)
        for layer, idx in selected:
            m = cache.m[layer]  # [B, T, d_mlp]
            v = m[..., idx]  # [B, T]
            # Take last 20 tokens (or all if shorter)
            v = v[:, -20:]
            sums[(layer, idx)] += float(v.sum().item())
            counts[(layer, idx)] += int(v.numel())

    return {k: sums[k] / max(1, counts[k]) for k in selected}


# ---------------------------------------------------------------------------
# Section 5.2 -- residual-stream offset delta_x
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_residual_offset(
    pre_model: GPT2WithHooks,
    post_model: GPT2WithHooks,
    input_ids_batch: list[torch.Tensor],
    layer: int,
) -> torch.Tensor:
    """delta_x^{layer-mid} = x_DPO^{layer-mid} - x_GPT2^{layer-mid}, mean over prompts.

    Returns the mean offset vector, a Tensor of shape [d_model].
    """
    diffs = []
    for ids in input_ids_batch:
        _, cache_pre = pre_model.forward_with_cache(ids)
        _, cache_post = post_model.forward_with_cache(ids)
        # x_mid at layer 'layer'
        x_pre = cache_pre.x_mid[layer].mean(dim=(0, 1))
        x_post = cache_post.x_mid[layer].mean(dim=(0, 1))
        diffs.append((x_post - x_pre).cpu())
    return torch.stack(diffs, dim=0).mean(dim=0)  # [d_model]
