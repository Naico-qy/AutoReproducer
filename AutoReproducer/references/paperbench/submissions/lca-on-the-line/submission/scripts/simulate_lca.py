"""Simulated-data illustration of LCA-on-the-Line (paper §3, Appendix C).

Reproduces Table 7. The data generation:
    z in {1, 2, 3, 4} (latent class), with class proximity
        root: ((1, 2), (3, 4))
    x | z=1 ~ N((1, 1, 0), I)
    x | z=2 ~ N((3, 17, 0), I)
    x | z=3 ~ N((15, 7, 0), I)
    x | z=4 ~ N((17, 21, 0), I)
where x_1 supports the hierarchy (transferable), x_2 is a confounder
(non-transferable) and x_3 is noise.

Two logistic regression models are trained:
    f -- on (x_1, x_3)  (transferable)
    g -- on (x_2, x_3)  (confounder)

For OOD, only x_1 and x_3 are observed (x_2 absent).  Per the paper, model
f attains *worse* ID Top1 but *better* OOD Top1 *and* lower ID LCA distance
— illustrating that LCA distance reflects feature transferability.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


# Pairwise LCA distance for the 4-class hierarchy: depth-based.
# Tree:           root
#                /    \
#            ( 1, 2 ) ( 3, 4 )
# Distance(i, i) = 0
# Distance(1, 2) = 2  (1 -> parent -> 2)
# Distance(3, 4) = 2
# Other pairs go through root: distance = 4
LCA_4 = np.array(
    [
        [0, 2, 4, 4],
        [2, 0, 4, 4],
        [4, 4, 0, 2],
        [4, 4, 2, 0],
    ],
    dtype=np.float32,
)


def sample_data(n_per_class: int, rng: np.random.Generator):
    means = np.array(
        [
            [1.0, 1.0, 0.0],
            [3.0, 17.0, 0.0],
            [15.0, 7.0, 0.0],
            [17.0, 21.0, 0.0],
        ]
    )
    X, y = [], []
    for k, mu in enumerate(means):
        Xk = rng.standard_normal((n_per_class, 3)) + mu
        X.append(Xk)
        y.append(np.full(n_per_class, k, dtype=np.int64))
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)


def lca_distance(preds: np.ndarray, targets: np.ndarray) -> float:
    """Mean LCA distance over misclassified examples (Eq. 2)."""
    mis = preds != targets
    if not np.any(mis):
        return 0.0
    return float(LCA_4[preds[mis], targets[mis]].mean())


def trial(seed: int, n_per_class: int = 2500):
    rng = np.random.default_rng(seed)
    Xtr, ytr = sample_data(n_per_class, rng)
    Xid, yid = sample_data(n_per_class, rng)
    Xood, yood = sample_data(n_per_class, rng)
    # OOD only observes x_1 and x_3 -- replace x_2 with zero (intervention).
    Xood[:, 1] = 0.0

    f = LogisticRegression(max_iter=1000).fit(Xtr[:, [0, 2]], ytr)
    g = LogisticRegression(max_iter=1000).fit(Xtr[:, [1, 2]], ytr)

    f_id = f.predict(Xid[:, [0, 2]])
    g_id = g.predict(Xid[:, [1, 2]])
    f_ood = f.predict(Xood[:, [0, 2]])
    g_ood = g.predict(Xood[:, [1, 2]])

    return {
        "f_id_err": float((f_id != yid).mean()),
        "g_id_err": float((g_id != yid).mean()),
        "f_ood_err": float((f_ood != yood).mean()),
        "g_ood_err": float((g_ood != yood).mean()),
        "f_id_lca": lca_distance(f_id, yid),
        "g_id_lca": lca_distance(g_id, yid),
    }


def main() -> None:
    rows = [trial(seed=i) for i in range(100)]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print("Simulation results (mean over 100 trials):")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
