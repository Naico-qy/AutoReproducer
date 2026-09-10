"""Graph Inversion (Webb et al. 2018), per addendum.md.

Given a directed Bayesian-net structure G (encoded as a square boolean
``M_E[i, j] = True  iff  j is a parent of i``) and a set of latent
variables Z (encoded as a 1-D bool ``M_C`` with True = observed), produce
the adjacency H that captures which nodes must attend to which under
``conditioning``-aware inversion.

Algorithm (addendum.md):
    1. Input: G, Z.
    2. J <- MORALIZE(G).      # undirected, parents joined.
    3. Unmark all vertices.
    4. H <- empty graph on V(G).
    5. S <- latent variables without latent parent in G.
    6. while S not empty:
    7.   v = argmin_{u in S} fill_size(u)            # min-fill heuristic.
    8.   add edges in J between unmarked neighbours of v.
    9.   make unmarked neighbours of v in J  v's parents in H.
   10.   mark v; remove from S.
   11.   for each unmarked latent child u of v in G:
   12.     if all parent latents of u in G are marked: add u to S.
   13. return H.

The final attention mask M_E^* = M_E_base ∨ adj(H).
"""

from __future__ import annotations

import numpy as np


def moralize(M_E: np.ndarray) -> np.ndarray:
    """Build the moral graph J from a directed adjacency M_E.

    M_E[i, j] = True iff j -> i. Moralization:
      - make undirected (J = M_E ∨ M_E.T)
      - connect parents: for every node i, fully connect rows of parents
        of i (i.e., for any pair (j, k) with M_E[i, j] = M_E[i, k] = True
        add J[j, k] = J[k, j] = True).
    """
    n = M_E.shape[0]
    J = (M_E | M_E.T).copy().astype(bool)
    for i in range(n):
        parents = np.where(M_E[i])[0]
        for a in parents:
            for b in parents:
                if a != b:
                    J[a, b] = True
                    J[b, a] = True
    np.fill_diagonal(J, False)
    return J


def _fill_size(J: np.ndarray, marked: np.ndarray, v: int) -> int:
    """How many edges adding v's clique would create (min-fill heuristic)."""
    nbrs = np.where(J[v] & ~marked)[0]
    add = 0
    for i, a in enumerate(nbrs):
        for b in nbrs[i + 1 :]:
            if not J[a, b]:
                add += 1
    return add


def graph_inversion(M_E: np.ndarray, M_C: np.ndarray) -> np.ndarray:
    """Return adjacency H (n×n bool) per the addendum's algorithm.

    H[i, j] = True iff j is a parent of i in the inverted (conditioned) graph.
    """
    M_E = M_E.astype(bool)
    M_C = M_C.astype(bool)
    n = M_E.shape[0]
    latent = ~M_C  # paper convention

    J = moralize(M_E)
    marked = np.zeros(n, dtype=bool)
    H = np.zeros((n, n), dtype=bool)

    # S: latents whose every parent in G is observed (i.e., no latent parent).
    def has_latent_parent(u: int) -> bool:
        parents = np.where(M_E[u])[0]
        return any(latent[p] for p in parents)

    S = [u for u in range(n) if latent[u] and not has_latent_parent(u)]

    while S:
        # Min-fill selection.
        v = min(S, key=lambda u: _fill_size(J, marked, u))
        S.remove(v)

        nbrs = np.where(J[v] & ~marked)[0]
        # Add edges in J between unmarked neighbours of v.
        for i, a in enumerate(nbrs):
            for b in nbrs[i + 1 :]:
                J[a, b] = True
                J[b, a] = True
        # Make unmarked neighbours of v parents of v in H.
        for a in nbrs:
            H[v, a] = True

        marked[v] = True

        # Add unmarked latent children of v whose latent parents are all marked.
        children = np.where(M_E[:, v])[0]
        for u in children:
            if marked[u] or not latent[u] or u in S:
                continue
            parent_lats = [p for p in np.where(M_E[u])[0] if latent[p]]
            if all(marked[p] for p in parent_lats):
                S.append(int(u))
    return H


def attention_mask_from_graph(M_E_base: np.ndarray, M_C: np.ndarray) -> np.ndarray:
    """Combine the base directed mask M_E_base with the inversion result H.

    Returns a boolean adjacency in the convention "True at (i, j) means
    token i is allowed to attend to token j" (i.e., the graph). To plug
    into PyTorch ``MultiheadAttention.attn_mask``, invert it to "block".
    """
    H = graph_inversion(M_E_base, M_C)
    return M_E_base.astype(bool) | H.astype(bool)
