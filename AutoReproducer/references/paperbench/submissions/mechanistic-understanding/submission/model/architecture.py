"""Core model components for reproducing Lee et al. 2024 (ICML 2024).

Paper: "A Mechanistic Understanding of Alignment Algorithms: A Case Study on
DPO and Toxicity"  --  https://github.com/ajyl/dpo_toxic
DPO loss reference: Rafailov et al. 2023, NeurIPS
    (verified via CrossRef in this submission; see ref_verify section in README)

This module implements:
  * GPT2WithHooks           -- GPT-2 medium wrapper that exposes the residual
                               stream and MLP key/value vectors needed by
                               Section 3 (toxic-vector extraction) and
                               Section 5 (post-DPO mechanistic analysis).
  * LinearToxicityProbe     -- the [d_model, 2] probe trained on Jigsaw.
                               Per addendum.md, W_toxic is a *matrix* and the
                               vector used for cosine similarity is W_toxic[:, 1]
                               (the "toxic" logit column).
  * extract_toxic_value_vectors -- selects the top-N MLP value vectors with
                               highest cosine similarity to the toxic probe
                               column (Section 3.1).
  * svd_decompose_toxic_vectors -- SVD of the d x N stacked toxic-value matrix
                               (per addendum: transpose of the N x d matrix
                               described in the paper text), returning
                               U_toxic[i] in R^d.
  * dpo_loss                -- the Section 4.1 DPO loss exactly as in
                               Rafailov et al. 2023.

Variable names mirror the paper where possible (W_toxic, x_l, m_i, k_i, v_i,
delta_x, etc.) so the code maps directly to the equations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ---------------------------------------------------------------------------
# 1. GPT-2 wrapper exposing residual streams + MLP weights
# ---------------------------------------------------------------------------
@dataclass
class HookedActivations:
    """Per-layer cached activations.

    Mirrors the notation in Section 2 of the paper.
        x_l      : residual stream at layer entry (post-prev-layer)
        x_l_mid  : residual stream after attention, before MLP
        m_l      : sigma(W_K^l x^l) -- the d_mlp coefficients (one per value vector)
        x_l_post : residual stream after the MLP block
    """

    x: dict  # int -> Tensor [B, T, d]
    x_mid: dict  # int -> Tensor [B, T, d]
    m: dict  # int -> Tensor [B, T, d_mlp]
    x_post: dict  # int -> Tensor [B, T, d]


class GPT2WithHooks(nn.Module):
    """Thin GPT-2 wrapper that records residual-stream and MLP activations.

    The base HF GPT-2 medium has 24 layers, d_model = 1024, d_mlp = 4096,
    GeLU activations -- consistent with paper Section 5 (which relies on
    GeLU's near-zero negative regime to explain the antipodal direction).
    """

    def __init__(self, name: str = "gpt2-medium", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.lm = GPT2LMHeadModel.from_pretrained(name, torch_dtype=dtype)
        self.tokenizer = GPT2Tokenizer.from_pretrained(name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = self.lm.config
        self.d_model = self.config.n_embd  # 1024 for gpt2-medium
        self.d_mlp = self.config.n_inner or 4 * self.d_model  # 4096
        self.n_layers = self.config.n_layer  # 24
        self._hooks = []

    # ---- weight accessors used in Section 3.1 / 5 -------------------------
    def mlp_W_K(self, layer: int) -> torch.Tensor:
        """Key matrix W_K^l, shape [d_mlp, d_model].

        In HF GPT-2 the MLP is implemented as Conv1D with weight shape
        [d_model, d_mlp].  We transpose it so each row is a key vector k_i^l
        in R^{d_model}, matching the paper's notation (Section 2).
        """
        return self.lm.transformer.h[layer].mlp.c_fc.weight.T  # [d_mlp, d_model]

    def mlp_W_V(self, layer: int) -> torch.Tensor:
        """Value matrix W_V^l, shape [d_mlp, d_model].

        Each row is a value vector v_i^l in R^{d_model}.  HF stores c_proj as
        [d_mlp, d_model] already, so no transpose needed.
        """
        return self.lm.transformer.h[layer].mlp.c_proj.weight  # [d_mlp, d_model]

    def value_vector(self, layer: int, idx: int) -> torch.Tensor:
        """v_idx^layer in R^{d_model}."""
        return self.mlp_W_V(layer)[idx].detach().clone()

    def key_vector(self, layer: int, idx: int) -> torch.Tensor:
        """k_idx^layer in R^{d_model}."""
        return self.mlp_W_K(layer)[idx].detach().clone()

    # ---- forward with caching --------------------------------------------
    @torch.no_grad()
    def forward_with_cache(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, HookedActivations]:
        """Run the model and capture per-layer activations.

        Returns
        -------
        logits : Tensor [B, T, |V|]
        cache  : HookedActivations  (one entry per layer 0..L-1)
        """
        cache_x: dict = {}
        cache_x_mid: dict = {}
        cache_m: dict = {}
        cache_x_post: dict = {}

        def make_pre_attn_hook(idx):
            def fn(_module, inputs):
                cache_x[idx] = inputs[0].detach()

            return fn

        def make_pre_mlp_hook(idx):
            def fn(_module, inputs):
                cache_x_mid[idx] = inputs[0].detach()

            return fn

        def make_post_block_hook(idx):
            def fn(_module, _inputs, outputs):
                out = outputs[0] if isinstance(outputs, tuple) else outputs
                cache_x_post[idx] = out.detach()

            return fn

        def make_mlp_act_hook(idx):
            # HF GPT-2: mlp.act runs *after* c_fc, *before* c_proj.
            # Its output is exactly m_i^l = sigma(W_K^l x^l) -- the d_mlp coeffs.
            def fn(_module, _inputs, outputs):
                cache_m[idx] = outputs.detach()

            return fn

        handles = []
        try:
            for i, block in enumerate(self.lm.transformer.h):
                handles.append(block.register_forward_pre_hook(make_pre_attn_hook(i)))
                handles.append(
                    block.mlp.register_forward_pre_hook(make_pre_mlp_hook(i))
                )
                handles.append(block.register_forward_hook(make_post_block_hook(i)))
                # GPT-2 MLP activation module name is "act"
                handles.append(
                    block.mlp.act.register_forward_hook(make_mlp_act_hook(i))
                )

            out = self.lm(input_ids=input_ids)
            logits = out.logits
        finally:
            for h in handles:
                h.remove()

        cache = HookedActivations(
            x=cache_x, x_mid=cache_x_mid, m=cache_m, x_post=cache_x_post
        )
        return logits, cache

    # ---- intervention used in Section 3.3 --------------------------------
    def install_residual_subtraction(
        self, layer: int, vector: torch.Tensor, alpha: float
    ):
        """Equation in Section 3.3:  x^{L-1} <- x^{L-1} - alpha * vector.

        Hook is removed by ``remove_interventions``.  ``layer`` is the residual
        stream layer index; we hook the post-block output of block ``layer-1``.
        """
        target_block = (
            self.lm.transformer.h[layer - 1] if layer > 0 else self.lm.transformer.wte
        )
        v = vector.to(next(self.lm.parameters()).device).detach()

        def hook(_module, _inputs, outputs):
            if isinstance(outputs, tuple):
                outputs = (outputs[0] - alpha * v,) + outputs[1:]
            else:
                outputs = outputs - alpha * v
            return outputs

        h = target_block.register_forward_hook(hook)
        self._hooks.append(h)

    def install_key_scaling(self, scaling: dict[tuple[int, int], float]):
        """Section 6 (un-align):  k_i^l <- scale * k_i^l for selected (l, i).

        Implemented in-place on a *clone* of c_fc.weight so the original DPO
        checkpoint is preserved if the user later calls ``remove_interventions``.
        """
        originals: list[tuple[int, torch.Tensor]] = []
        for (layer, idx), s in scaling.items():
            W = self.lm.transformer.h[layer].mlp.c_fc.weight  # [d_model, d_mlp]
            originals.append((layer, W.detach().clone()))
            with torch.no_grad():
                W[:, idx] *= s
        # store originals so we can restore
        self._key_scale_backup = originals

    def remove_interventions(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        if hasattr(self, "_key_scale_backup"):
            for layer, W_orig in self._key_scale_backup:
                with torch.no_grad():
                    self.lm.transformer.h[layer].mlp.c_fc.weight.copy_(W_orig)
            del self._key_scale_backup


# ---------------------------------------------------------------------------
# 2. Linear toxicity probe (Section 3.1)
# ---------------------------------------------------------------------------
class LinearToxicityProbe(nn.Module):
    """The W_toxic probe, trained on Jigsaw last-layer mean-pooled hidden states.

    Per addendum.md, W_toxic is *not* a vector but the [d_model, 2] weight
    matrix of a binary softmax classifier:
        P(toxic | x_bar^{L-1}) = softmax(W_toxic^T x_bar^{L-1})
    The "probe vector" used for cosine similarity in Section 3.1 is the
    second column W_toxic[:, 1] (the toxic logit direction).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, 2, bias=True)

    def forward(self, x_bar: torch.Tensor) -> torch.Tensor:
        # x_bar: [B, d_model] -- mean-pooled residual stream from layer L-1
        return self.fc(x_bar)

    @property
    def W_toxic(self) -> torch.Tensor:
        """Returns the [d_model, 2] matrix.  Use ``.toxic_direction`` for the vector."""
        return (
            self.fc.weight.T
        )  # nn.Linear stores [out, in], so transpose -> [in, out] = [d_model, 2]

    @property
    def toxic_direction(self) -> torch.Tensor:
        """W_toxic[:, 1] -- the d_model vector used in Section 3.1."""
        return self.W_toxic[:, 1].detach().clone()


# ---------------------------------------------------------------------------
# 3. Section 3.1: extract toxic MLP value vectors by cosine similarity
# ---------------------------------------------------------------------------
def extract_toxic_value_vectors(
    model: GPT2WithHooks,
    probe: LinearToxicityProbe,
    n_top: int = 128,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Section 3.1: select the top-N MLP value vectors v_i^l with the highest
    cosine similarity to W_toxic[:, 1].

    Returns
    -------
    V_toxic   : Tensor [N, d_model]  -- stacked toxic value vectors
    indices   : list of (layer, idx) for each row of V_toxic
    """
    probe_dir = probe.toxic_direction  # [d_model]
    probe_dir = probe_dir / (probe_dir.norm() + 1e-8)

    cos_scores = []
    coords = []
    for layer in range(model.n_layers):
        Wv = model.mlp_W_V(layer).detach()  # [d_mlp, d_model]
        Wv_norm = Wv / (Wv.norm(dim=1, keepdim=True) + 1e-8)
        sims = Wv_norm @ probe_dir.to(Wv_norm.device)  # [d_mlp]
        for idx in range(sims.shape[0]):
            cos_scores.append(sims[idx].item())
            coords.append((layer, idx))

    # top-N by descending cosine similarity (most aligned with toxic direction)
    order = sorted(range(len(cos_scores)), key=lambda i: -cos_scores[i])[:n_top]
    selected = [coords[i] for i in order]

    rows = []
    for layer, idx in selected:
        rows.append(model.value_vector(layer, idx))
    V_toxic = torch.stack(rows, dim=0)  # [N, d_model]
    return V_toxic, selected


# ---------------------------------------------------------------------------
# 4. Section 3.1: SVD decomposition of the toxic value-vector matrix
# ---------------------------------------------------------------------------
def svd_decompose_toxic_vectors(V_toxic: torch.Tensor) -> torch.Tensor:
    """Per addendum.md: the paper text says "SVD on N x d", but the authors
    actually mean SVD on the *transpose* (d x N) so that the singular vectors
    in U live in R^d (the model's hidden space).

    Parameters
    ----------
    V_toxic : [N, d_model]  -- stacked toxic value vectors

    Returns
    -------
    U_toxic : [d_model, k]  -- columns are the SVD.U_toxic[i] basis vectors
                                that span the toxic representation subspace.
    """
    # Transpose first  ->  [d_model, N]
    M = V_toxic.T.contiguous().to(torch.float32)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    # U: [d_model, min(d_model, N)],  S: [min(...)], Vh: [min(...), N]
    return U  # column i is SVD.U_toxic[i]  (an R^d vector)


# ---------------------------------------------------------------------------
# 5. Section 4.1: DPO loss (Rafailov et al. 2023)
# ---------------------------------------------------------------------------
def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equation in Section 4.1:

        L_DPO = -E[ log sigma( beta * log P  -  beta * log N ) ]

    where  P = pi_theta(y+|w) / pi_ref(y+|w),  N = pi_theta(y-|w) / pi_ref(y-|w).

    All inputs are *summed* log-probabilities of the chosen / rejected
    completions (one scalar per example), as is standard for DPO.

    Returns
    -------
    loss            : scalar  L_DPO
    chosen_rewards  : beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards: beta * (policy_rejected_logps - ref_rejected_logps)
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    return loss, chosen_rewards, rejected_rewards
