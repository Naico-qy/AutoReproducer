"""LexiFlow: randomized direct search with lexicographic preferences.

This implements Algorithm 2 of the paper (Appendix A) — adapted from
LexiFlow (Zhang et al., 2023b/c). For RCS the variants of the algorithm:
  - the input target list is dropped (not needed),
  - the compromise epsilon is interpreted as a *relative* tolerance.

The mask `m` is maintained as a continuous vector in [-1, 1]^n (per the paper,
just before discretisation: "the value of m less than -1 becomes -1 and the
value greater than 1 becomes 1. Then during discretization, m in [-1, 0) will
be projected to 0, and m in [0, 1] will be projected to 1.").

Calls a user-supplied `evaluate(mask01) -> (f1, f2)` to evaluate any candidate.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from .lexico import History, ObjVec, is_better, lex_lt


def discretize(m_cont: np.ndarray) -> np.ndarray:
    """Clip to [-1, 1], then project: [-1, 0) -> 0, [0, 1] -> 1."""
    m = np.clip(m_cont, -1.0, 1.0)
    return (m >= 0.0).astype(np.uint8)


def sample_unit(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample uniformly on the unit sphere S^{n-1}."""
    u = rng.standard_normal(n)
    norm = np.linalg.norm(u) + 1e-12
    return u / norm


def lexiflow_search(
    n: int,
    evaluate: Callable[[np.ndarray], ObjVec],
    *,
    T: int,
    epsilon: float,
    delta_init: float = 0.5,
    delta_lower: float = 1e-3,
    init_mask01: np.ndarray | None = None,
    seed: int = 0,
    log_every: int = 50,
    logger=print,
) -> Tuple[np.ndarray, ObjVec]:
    """Run LexiFlow for `T` outer iterations and return (best_mask01, best_F).

    Parameters
    ----------
    n          : length of the binary mask.
    evaluate   : function taking a 0/1 mask of length n and returning (f1, f2).
    T          : number of outer iterations.
    epsilon    : relative compromise of f1 (Definition 1).
    delta_init : initial step size in LexiFlow.
    delta_lower: random-restart threshold for delta.
    init_mask01: optional initial 0/1 mask; default: random {0,1}.
    seed       : RNG seed.
    """
    rng = np.random.default_rng(seed)

    if init_mask01 is None:
        m01 = rng.integers(0, 2, size=n).astype(np.uint8)
    else:
        m01 = init_mask01.astype(np.uint8).copy()

    # represent in continuous space (paper: -1 -> 0, +1 -> 1 mid-line)
    m_cont = np.where(m01 == 1, 0.5, -0.5).astype(np.float64)

    F_cur = evaluate(m01)
    F_best = F_cur
    m_best = m01.copy()

    hist = History(fs=[F_cur], epsilon=epsilon)

    delta = delta_init
    t_prime = 0
    e = 0
    r = 0
    n_dim = n

    for t in range(T):
        F_H = hist.thresholds()
        u = sample_unit(n_dim, rng)

        m_plus = discretize(m_cont + delta * u)
        F_plus = evaluate(m_plus)
        hist.push(F_plus)

        accepted = False
        if is_better(F_plus, F_cur, F_H):
            m_cont = np.clip(m_cont + delta * u, -1.0, 1.0)
            m01 = m_plus
            F_cur = F_plus
            t_prime = t
            accepted = True
            # update incumbent best (paper compares best vs new)
            F_H_b = hist.thresholds()
            if is_better(F_plus, F_best, F_H_b) or (lex_lt(F_plus, F_best)):
                F_best = F_plus
                m_best = m_plus.copy()
        else:
            m_minus = discretize(m_cont - delta * u)
            F_minus = evaluate(m_minus)
            hist.push(F_minus)
            if is_better(F_minus, F_cur, F_H):
                m_cont = np.clip(m_cont - delta * u, -1.0, 1.0)
                m01 = m_minus
                F_cur = F_minus
                t_prime = t
                accepted = True
                F_H_b = hist.thresholds()
                if is_better(F_minus, F_best, F_H_b) or lex_lt(F_minus, F_best):
                    F_best = F_minus
                    m_best = m_minus.copy()

        if not accepted:
            e += 1

        # adapt delta — every 2^{n-1} unsuccessful steps, shrink delta. We use
        # min(2**(n_dim-1), 64) to avoid astronomical waiting on n>>1.
        threshold_e = min(2 ** max(int(np.log2(max(n_dim, 2))) - 1, 0), 64)
        if e >= threshold_e:
            e = 0
            delta = delta * np.sqrt((t_prime + 1) / (t + 1))

        if delta < delta_lower:
            # random restart
            r += 1
            m_cont = rng.normal(loc=0.0, scale=1.0, size=n_dim)
            m_cont = np.clip(m_cont, -1.0, 1.0)
            m01 = discretize(m_cont)
            F_cur = evaluate(m01)
            hist.push(F_cur)
            delta = delta_init + r

        if (t + 1) % log_every == 0 and logger is not None:
            logger(
                f"[LexiFlow] iter={t + 1}/{T} f1={F_cur[0]:.4f} f2={int(F_cur[1])} "
                f"best=({F_best[0]:.4f}, {int(F_best[1])}) delta={delta:.4f}"
            )

    return m_best, F_best
