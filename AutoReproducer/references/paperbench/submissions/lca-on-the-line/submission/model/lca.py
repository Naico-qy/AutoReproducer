"""Lowest Common Ancestor (LCA) distance computation.

This module is the core of the paper "LCA-on-the-Line: Benchmarking
Out-of-Distribution Generalization with Class Taxonomies" (Shi et al.,
ICML 2024). It implements:

  * Hierarchy abstractions over either WordNet (for ImageNet) or a latent
    hierarchy built via K-means clustering on per-class mean features
    (paper §4.3.1, Appendix E.1).
  * The LCA distance D_LCA(y', y) (Eq. 1 of the paper):
        D_LCA(y', y) := f(y) - f(N_LCA(y, y'))
    with two scoring functions (paper Appendix D.2):
        - tree depth f = P(.)              (used for linear probing)
        - information content f = I(.)     (used for measurement; addendum
          mandates I(y) = -log p(y) = log|L| - log|L(y)| as in Valmadre 2022)
  * The dataset-level mistake-severity score (Eq. 2):
        D_LCA(model, M) = (1/n) sum_i D_LCA(yhat_i, y_i)  if y_i != yhat_i
  * The Expected LCA distance (ELCA, Appendix D.3):
        D_ELCA(model, M) = (1/(nK)) sum_i sum_k phat_{k,i} * D_LCA(k, y_i)
  * The pairwise LCA distance matrix M_LCA (Appendix E.2) with
    `process_lca_matrix` from the addendum that inverts the matrix when the
    hierarchy is latent (NOT WordNet) and applies temperature + MinMax.

Reference (verified via paper_search/CrossRef):
    Bertinetto et al. "Making Better Mistakes: Leveraging Class Hierarchies
    with Deep Networks." CVPR 2020. (verified)
    Valmadre, J. "Hierarchical classification at multiple operating points."
    arXiv:2210.10929, 2022. (information content definition).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Hierarchy abstractions
# ---------------------------------------------------------------------------


@dataclass
class Hierarchy:
    """A simple tree-shaped class hierarchy.

    parent[node]  -> parent_node      (root has parent == None or -1)
    leaves        -> ordered list of K leaf node ids that correspond to the
                     class indices [0, ..., K-1]
    """

    parents: Dict[int, Optional[int]]
    leaves: List[int]

    def __post_init__(self) -> None:
        # Pre-compute depths and ancestor sets for fast lookup.
        self._depth_cache: Dict[int, int] = {}
        self._ancestors_cache: Dict[int, List[int]] = {}
        for n in list(self.parents):
            self.depth(n)
            self.ancestors(n)
        # Information content: I(y) = log|L| - log|L(y)|
        self._info_cache: Dict[int, float] = self._compute_information()

    # -- structural helpers -------------------------------------------------
    def depth(self, node: int) -> int:
        if node in self._depth_cache:
            return self._depth_cache[node]
        d, cur = 0, node
        while self.parents.get(cur) is not None and self.parents[cur] != cur:
            cur = self.parents[cur]
            d += 1
        self._depth_cache[node] = d
        return d

    def ancestors(self, node: int) -> List[int]:
        """Return [node, parent, grandparent, ..., root]."""
        if node in self._ancestors_cache:
            return self._ancestors_cache[node]
        chain, cur = [node], node
        while self.parents.get(cur) is not None and self.parents[cur] != cur:
            cur = self.parents[cur]
            chain.append(cur)
        self._ancestors_cache[node] = chain
        return chain

    def lca_node(self, a: int, b: int) -> int:
        """Lowest common ancestor of two nodes a, b."""
        if a == b:
            return a
        anc_a = set(self.ancestors(a))
        for n in self.ancestors(b):
            if n in anc_a:
                return n
        # Fallback to root
        return self.ancestors(a)[-1]

    # -- score functions f(.) ----------------------------------------------
    def tree_depth_score(self, node: int) -> int:
        """f(node) = P(node) = depth in the tree."""
        return self.depth(node)

    def information_score(self, node: int) -> float:
        """f(node) = I(node) = -log p(node) where p is uniform over leaves.

        Addendum specifies: I(y) = -log p(y) = log|L| - log|L(y)|.
        """
        return self._info_cache.get(node, 0.0)

    def _compute_information(self) -> Dict[int, float]:
        """Compute information content for every node assuming a uniform
        distribution over leaves."""
        leaf_set = set(self.leaves)
        total_leaves = max(len(self.leaves), 1)
        # leaves_under[node] = number of leaves descending from `node`
        leaves_under: Dict[int, int] = {n: 0 for n in self.parents}
        for leaf in self.leaves:
            for anc in self.ancestors(leaf):
                leaves_under[anc] = leaves_under.get(anc, 0) + 1
        # Avoid double counting at the leaf itself
        for n in list(leaves_under):
            if n in leaf_set:
                # re-count: a leaf has exactly 1 leaf descendant (itself)
                pass
        info: Dict[int, float] = {}
        log_L = math.log(total_leaves)
        for n, count in leaves_under.items():
            count = max(int(count), 1)
            # I(n) = log|L| - log|L(n)|
            info[n] = log_L - math.log(count)
        return info


# ---------------------------------------------------------------------------
# WordNet hierarchy for ImageNet
# ---------------------------------------------------------------------------


class WordNetHierarchy(Hierarchy):
    """ImageNet-1k WordNet hierarchy, loaded from the CSV provided by the
    `hiercls` repository (https://github.com/jvlmdr/hiercls), as documented in
    the paper's addendum.

    The CSV stores each row as a path of WordNet synsets from root to a leaf
    node; we collapse these into a tree and remember the order of leaf nodes
    so that class index k corresponds to ImageNet leaf #k (caller-supplied).
    """

    def __init__(
        self, csv_path: str, leaf_order: Optional[Sequence[str]] = None
    ) -> None:
        parents: Dict[str, Optional[str]] = {}
        all_nodes: List[str] = []
        leaves_in_order: List[str] = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                # Skip blank / commented rows
                row = [r.strip() for r in row if r.strip()]
                if not row:
                    continue
                # row = [root, ..., leaf]
                for i, node in enumerate(row):
                    if node not in parents:
                        parents[node] = row[i - 1] if i > 0 else None
                        all_nodes.append(node)
                # last entry is a leaf
                leaf = row[-1]
                if leaf not in leaves_in_order:
                    leaves_in_order.append(leaf)

        # Map string node ids to ints
        ids = {name: idx for idx, name in enumerate(all_nodes)}
        int_parents: Dict[int, Optional[int]] = {}
        for name, par in parents.items():
            int_parents[ids[name]] = ids[par] if par is not None else None
        if leaf_order is not None:
            leaf_ids = [ids[name] for name in leaf_order if name in ids]
        else:
            leaf_ids = [ids[name] for name in leaves_in_order]
        super().__init__(parents=int_parents, leaves=leaf_ids)
        self._name_to_id = ids
        self._id_to_name = {v: k for k, v in ids.items()}

    def class_to_node(self, class_idx: int) -> int:
        return self.leaves[class_idx]


# ---------------------------------------------------------------------------
# Latent hierarchy via K-means clustering (paper §4.3.1, Appendix E.1)
# ---------------------------------------------------------------------------


class KMeansLatentHierarchy:
    """Construct a latent class hierarchy by hierarchically clustering the
    per-class mean feature vectors with K-means (Algorithm in Appendix E.1).

    For a 1000-class dataset, the paper uses 9 levels (2^9 < 1000), running
    K-means independently at each level with K = 2, 4, 8, ..., 512 cluster
    centers. The pairwise LCA height between two classes is the smallest
    cluster level at which both classes share a cluster (addendum).
    """

    def __init__(self, num_levels: int = 9, random_state: int = 0) -> None:
        self.num_levels = num_levels
        self.random_state = random_state
        self.lca_matrix: Optional[np.ndarray] = None
        self.cluster_assignments: List[np.ndarray] = []

    def fit(self, class_features: np.ndarray) -> "KMeansLatentHierarchy":
        """Fit hierarchical clustering on per-class mean features.

        Args:
            class_features: shape (K, D) array of per-class average features.
        Returns: self.
        """
        K = class_features.shape[0]
        # Run K-means at each granularity level i in {1..num_levels}, K_i = 2^i
        assignments: List[np.ndarray] = []
        for i in range(1, self.num_levels + 1):
            n_clusters = min(2**i, K)
            km = KMeans(
                n_clusters=n_clusters, n_init=10, random_state=self.random_state
            )
            labels = km.fit_predict(class_features)
            assignments.append(labels)
        self.cluster_assignments = assignments

        # Compute pairwise LCA height: cluster level at which both classes
        # first share a cluster.  By definition, all classes share a base
        # cluster at level (num_levels + 1) (paper E.1).
        height = np.full((K, K), self.num_levels + 1, dtype=np.float32)
        # iterate from coarsest (level 1) to finest (level num_levels)
        for level, labels in enumerate(assignments, start=1):
            same = labels[:, None] == labels[None, :]
            # If they share a cluster at this level (which is finer), update.
            # We want the smallest level at which they share -> greedy update.
            mask = same & (height == self.num_levels + 1)
            # Actually the paper says: "the cluster level at which a pair of
            # classes first share a cluster is the pairwise LCA height" — the
            # FIRST level (i.e. the FINEST that still groups them).  Smaller
            # level number = coarser cluster.  We thus walk from finest to
            # coarsest and keep the smallest height = first observed.
            height[mask] = self.num_levels + 1 - level
        # diagonal must be zero (sanity check from addendum)
        np.fill_diagonal(height, 0.0)
        self.lca_matrix = height
        return self

    def matrix(self) -> np.ndarray:
        if self.lca_matrix is None:
            raise RuntimeError("Call .fit() first to build the latent hierarchy.")
        return self.lca_matrix


# ---------------------------------------------------------------------------
# Sample-level LCA distances (paper Eq. 1, 2)
# ---------------------------------------------------------------------------


def lca_distance(
    pred: int,
    target: int,
    hier: Hierarchy,
    score: str = "information",
) -> float:
    """Compute D_LCA(y', y) for a single (prediction, target) pair.

    Args:
        pred: predicted class index.
        target: ground-truth class index.
        hier: a Hierarchy instance providing leaf ordering.
        score: "information" (default; used for measurement) or "depth"
            (used for linear-probing soft loss as in paper Appendix D.2).
    """
    if pred == target:
        return 0.0
    y_node = hier.leaves[target]
    yhat_node = hier.leaves[pred]
    n_lca = hier.lca_node(y_node, yhat_node)
    if score == "depth":
        # Symmetric tree-depth distance (Appendix D.2)
        return float(
            (hier.tree_depth_score(y_node) - hier.tree_depth_score(n_lca))
            + (hier.tree_depth_score(yhat_node) - hier.tree_depth_score(n_lca))
        )
    if score == "information":
        # Eq. 1 with f = I(.)
        return float(hier.information_score(y_node) - hier.information_score(n_lca))
    raise ValueError(f"Unknown score function: {score}")


def lca_distance_dataset(
    preds: torch.Tensor,
    targets: torch.Tensor,
    hier: Hierarchy,
    score: str = "information",
    misclassified_only: bool = True,
) -> float:
    """Implements Eq. 2 of the paper.

    D_LCA(model, M) = (1/n) sum_i D_LCA(yhat_i, y_i) over y_i != yhat_i
    when `misclassified_only=True` (default, matches the paper's wording),
    or over all samples otherwise.
    """
    preds_np = preds.detach().cpu().numpy().astype(int)
    targets_np = targets.detach().cpu().numpy().astype(int)
    n = len(preds_np)
    if n == 0:
        return 0.0
    total = 0.0
    counted = 0
    for p, t in zip(preds_np, targets_np):
        if misclassified_only and p == t:
            continue
        total += lca_distance(int(p), int(t), hier, score=score)
        counted += 1
    denom = counted if misclassified_only else n
    return total / max(denom, 1)


def expected_lca_distance(
    probs: torch.Tensor,
    targets: torch.Tensor,
    lca_pairwise: torch.Tensor,
) -> float:
    """Implements Eq. for ELCA in Appendix D.3.

        D_ELCA(model, M) = (1/(nK)) sum_i sum_k phat_{k,i} * D_LCA(k, y_i)

    Args:
        probs: (n, K) tensor of softmax probabilities over K classes.
        targets: (n,) tensor of ground-truth indices.
        lca_pairwise: (K, K) tensor of pairwise LCA distances; entry (i, j)
            should be D_LCA(i, j) (raw, NOT normalized).
    """
    if probs.dim() != 2:
        raise ValueError("probs must be (n, K)")
    n, K = probs.shape
    rows = lca_pairwise[targets]  # (n, K)
    weighted = probs * rows  # (n, K)
    return float(weighted.sum().item()) / float(n * K)


# ---------------------------------------------------------------------------
# Pairwise LCA distance matrix (paper Appendix E.2 + addendum)
# ---------------------------------------------------------------------------


def build_lca_matrix(
    hier: Hierarchy,
    score: str = "depth",
) -> np.ndarray:
    """Compute the K x K pairwise LCA distance matrix M[i,j] = D_LCA(i, j).

    The diagonal is zero (sanity check from addendum: distance to self == 0).
    """
    K = len(hier.leaves)
    M = np.zeros((K, K), dtype=np.float32)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            M[i, j] = lca_distance(i, j, hier, score=score)
    # sanity check: diagonal is zero
    assert np.all(np.diag(M) == 0), "LCA distance matrix must have zero diagonal"
    return M


def process_lca_matrix(
    lca_matrix_raw: np.ndarray,
    tree_prefix: str = "WordNet",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Reproduce `process_lca_matrix` from the paper's addendum.

    For latent hierarchies (tree_prefix != "WordNet"), the matrix is INVERTED
    (max - x) before scaling — addendum verbatim. For WordNet, no inversion.
    Then apply temperature exponent and MinMax scaling.

    Returns a torch.Tensor M_LCA in [0, 1] suitable for use in soft-label loss
    construction (Algorithm 1, Appendix E.2). After this transform, the
    INVERTED LCA distance matrix (1 - M_LCA) should have ones on its diagonal
    (sanity check from addendum).
    """
    if lca_matrix_raw is None:
        return None
    if tree_prefix != "WordNet":
        result = float(np.max(lca_matrix_raw)) - lca_matrix_raw
    else:
        result = lca_matrix_raw.copy()
    result = result**temperature
    scaler = MinMaxScaler()
    result = scaler.fit_transform(result)
    # Sanity check: after MinMax, the WordNet matrix has zero diagonal
    # because all D_LCA(i, i) = 0 are mapped to 0 (the column min).  The
    # inverted matrix (1 - M_LCA) thus has ones on the diagonal as the
    # addendum requires.
    return torch.from_numpy(result.astype(np.float32))


def per_class_mean_features(
    features: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Return (K, D) tensor of per-class average features.

    Used both for K-means latent hierarchy construction (Appendix E.1) and as
    the input to LCA-soft-label experiments. Addendum specifies that features
    M(X) are taken from the last hidden layer before the FC classifier.
    """
    K = num_classes
    D = features.shape[1]
    sums = torch.zeros((K, D), dtype=features.dtype)
    counts = torch.zeros(K, dtype=torch.long)
    for f, t in zip(features, targets):
        sums[t] += f
        counts[t] += 1
    counts = counts.clamp_min_(1).unsqueeze(-1).to(features.dtype)
    return sums / counts
