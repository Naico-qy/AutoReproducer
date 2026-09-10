"""Simformer architecture (Gloeckler et al. ICML 2024, §3).

The Simformer is a probabilistic diffusion model whose score s_φ(x̂_t, t)
is parameterized by a transformer over a sequence of *tokens*, one per
variable in the joint x̂ = (θ, x). Each token (per Tokenizer §3.1 and the
addendum) is the concatenation of:

   identifier_embedding ⊕ value_embedding ⊕ [optional metadata] ⊕ condition_embedding

where:
- identifier_embedding: a learnable embedding indexed by variable position id;
- value_embedding: the scalar value broadcast to the embedding dim
  (i.e., ``[v, v, ..., v]`` of length ``embed_dim``) -- per addendum.md
  "to embed the value 1 to desired dimensionality N, we would have a vector
  [1, 1, ...., 1] of length N";
- condition_embedding: a learnable vector for "True" (conditioned), zeros
  for "False" (latent) -- per addendum.md.

A Gaussian Fourier embedding of the diffusion time t is added to the output
of every feed-forward block (per addendum.md "a linear projection is added
to the output of each feed-forward block").

For nonparametric (function-valued) parameters, the identifier is the sum of
a shared embedding vector and a random Fourier embedding of the index value
(see paper §3.1 & §4.3).
"""

from __future__ import annotations

import math
import torch
from torch import nn, Tensor


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class GaussianFourierEmbedding(nn.Module):
    """Random Gaussian Fourier features of a scalar input (e.g., diffusion time)."""

    def __init__(self, dim: int, scale: float = 16.0) -> None:
        super().__init__()
        # B has shape (dim/2,), frozen
        self.register_buffer("B", torch.randn(dim // 2) * scale)

    def forward(self, t: Tensor) -> Tensor:
        # t: (B,) -> (B, dim)
        proj = 2.0 * math.pi * t.unsqueeze(-1) * self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class IndexFourierEmbedding(nn.Module):
    """Fourier embedding for continuous index values (used for ∞-dim params)."""

    def __init__(self, dim: int, scale: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("B", torch.randn(dim // 2) * scale)

    def forward(self, idx: Tensor) -> Tensor:
        proj = 2.0 * math.pi * idx.unsqueeze(-1) * self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# ---------------------------------------------------------------------------
# Tokenizer (§3.1, addendum.md "Tokenization")
# ---------------------------------------------------------------------------


class Tokenizer(nn.Module):
    """Build per-variable tokens for the Simformer.

    For variable i with value v_i and condition flag m_i:
        token_i = id_embed[i] ⊕ broadcast(v_i, D) ⊕ cond_embed(m_i)
    Output dim = 3 * D.

    A final linear projection maps to ``embedding_dim`` so the transformer
    width is ``D = embedding_dim``.
    """

    def __init__(self, num_variables: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_variables = num_variables
        self.embedding_dim = embedding_dim

        # Variable identifier embedding (one per slot)
        self.id_embed = nn.Embedding(num_variables, embedding_dim)
        # Condition state embedding: True -> learnable vec; False -> zeros
        self.cond_true = nn.Parameter(torch.randn(embedding_dim) * 0.02)
        # Final projection: concat(id, val, cond) -> embedding_dim
        self.proj = nn.Linear(3 * embedding_dim, embedding_dim)

    def forward(self, values: Tensor, condition_mask: Tensor) -> Tensor:
        """values: (B, N), condition_mask: (B, N) bool/0-1."""
        B, N = values.shape
        device = values.device
        # Identifier embedding
        ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        id_emb = self.id_embed(ids)  # (B, N, D)
        # Value embedding: broadcast scalar to the embedding dim
        val_emb = values.unsqueeze(-1).expand(-1, -1, self.embedding_dim)  # (B, N, D)
        # Condition embedding: True -> learnable vector; False -> zeros
        cond = condition_mask.float().unsqueeze(-1)
        cond_emb = cond * self.cond_true  # (B, N, D)
        token = torch.cat([id_emb, val_emb, cond_emb], dim=-1)
        return self.proj(token)  # (B, N, D)


# ---------------------------------------------------------------------------
# Transformer block with diffusion-time conditioning
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Standard transformer block (pre-norm) with attention-mask support
    plus an additive linear projection of a diffusion-time embedding into
    the output of the feed-forward network (addendum.md).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        fourier_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        # Linear projection of the diffusion-time Fourier embedding,
        # added to the FFN output (per addendum.md).
        self.t_proj = nn.Linear(fourier_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, t_emb: Tensor, attn_mask: Tensor | None = None
    ) -> Tensor:
        # x: (B, N, D), t_emb: (B, fourier_dim), attn_mask: (B, N, N) or (N, N)
        h = self.norm1(x)
        # nn.MultiheadAttention expects ``attn_mask`` as additive bias if float
        # or boolean mask where True means "block".
        if attn_mask is not None and attn_mask.dim() == 3:
            # Expand for heads: (B, N, N) -> (B*H, N, N)
            B, N, _ = attn_mask.shape
            H = self.attn.num_heads
            attn_mask = (
                attn_mask.unsqueeze(1).expand(-1, H, -1, -1).reshape(B * H, N, N)
            )
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(attn_out)
        h = self.norm2(x)
        ff_out = self.ff(h)
        # Inject diffusion-time embedding: broadcast over the sequence axis.
        ff_out = ff_out + self.t_proj(t_emb).unsqueeze(1)
        x = x + self.dropout(ff_out)
        return x


# ---------------------------------------------------------------------------
# Full Simformer score model
# ---------------------------------------------------------------------------


class Simformer(nn.Module):
    """Score network s_φ^{M_E}(x̂_t, t) for the Simformer.

    Inputs
    ------
    values            : (B, N) the (partially noised) joint x̂_t^{M_C}
    condition_mask    : (B, N) M_C (True -> conditioned/observed).
    t                 : (B,)   diffusion time in [eps, T].
    attention_mask    : (N, N) or (B, N, N) bool tensor M_E.
                       True at (i, j) means token j is **blocked** from
                       attending to token i (PyTorch convention). We follow
                       the additive boolean convention. ``None`` means dense
                       attention.

    Output
    ------
    score             : (B, N) estimated ∇ log p_t(x̂_t).
    """

    def __init__(
        self,
        num_variables: int,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 6,
        ff_dim: int = 256,
        fourier_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_variables = num_variables
        self.embedding_dim = embedding_dim

        self.tokenizer = Tokenizer(num_variables, embedding_dim)
        self.time_embed = GaussianFourierEmbedding(fourier_dim, scale=16.0)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embedding_dim, num_heads, ff_dim, fourier_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embedding_dim)
        # Score head: project each token to a scalar score component.
        self.score_head = nn.Linear(embedding_dim, 1)

    def forward(
        self,
        values: Tensor,
        condition_mask: Tensor,
        t: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.tokenizer(values, condition_mask)  # (B, N, D)
        t_emb = self.time_embed(t)  # (B, fourier_dim)
        for block in self.blocks:
            x = block(x, t_emb, attention_mask)
        x = self.norm(x)
        score = self.score_head(x).squeeze(-1)  # (B, N)
        return score
