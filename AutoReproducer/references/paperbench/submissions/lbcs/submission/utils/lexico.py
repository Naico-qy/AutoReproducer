"""Practical lexicographic relations for LBCS (Appendix A of the paper).

Given two ordered objective vectors F(m) = [f1(m), f2(m)] and a "history"
threshold vector F_H = [f1*_tilde, f2*_tilde], we define:

    F(m)  =_(F_H)  F(m')   iff  for all i:  f_i(m) == f_i(m')
                                            OR (f_i(m) <= f_i*_tilde
                                                AND f_i(m') <= f_i*_tilde)

    F(m)  <_(F_H)  F(m')   iff  exists i:  f_i(m) < f_i(m')
                                            AND f_i(m') > f_i*_tilde
                                            AND F_{i-1}(m) =_(F_H) F_{i-1}(m')

    F(m)  <=_(F_H) F(m')   iff  F(m) <_(F_H) F(m')  OR  F(m) =_(F_H) F(m')

Plus the strict/exact relations from Definition 1 (without thresholds):

    F(m)  ==  F(m')  iff  forall i: f_i(m) == f_i(m')
    F(m)   <  F(m')  iff  exists i: f_i(m) < f_i(m')
                           and forall i' < i: f_i'(m) == f_i'(m')

Threshold computation (eq. 14 / 15 of the paper):

    f1*_hat = inf over history of f1(m)
    f1*_tilde = f1*_hat * (1 + epsilon)
    f2*_hat = inf over { m in history with f1(m) <= f1*_tilde } of f2(m)
    f2*_tilde = f2*_hat
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


# ---------- F(m): tuple of objective values ----------
ObjVec = Tuple[float, float]  # (f1, f2)


@dataclass
class History:
    """History of evaluated points, used to compute the F_H thresholds."""

    fs: List[ObjVec]
    epsilon: float

    def push(self, F: ObjVec) -> None:
        self.fs.append(F)

    def thresholds(self) -> ObjVec:
        if not self.fs:
            return (float("inf"), float("inf"))
        f1_hat = min(f[0] for f in self.fs)
        f1_tilde = f1_hat * (1.0 + self.epsilon)
        candidates = [f for f in self.fs if f[0] <= f1_tilde]
        f2_hat = (
            min(f[1] for f in candidates) if candidates else min(f[1] for f in self.fs)
        )
        f2_tilde = f2_hat
        return (f1_tilde, f2_tilde)


# ---------- exact lexicographic relations (Definition 1) ----------
def lex_eq(a: ObjVec, b: ObjVec) -> bool:
    return all(a[i] == b[i] for i in range(len(a)))


def lex_lt(a: ObjVec, b: ObjVec) -> bool:
    """Strict lexicographic less-than: exists i s.t. a_i < b_i and a_j == b_j for all j<i."""
    for i in range(len(a)):
        if a[i] < b[i]:
            return all(a[j] == b[j] for j in range(i))
        if a[i] > b[i]:
            return False
    return False


def lex_le(a: ObjVec, b: ObjVec) -> bool:
    return lex_lt(a, b) or lex_eq(a, b)


# ---------- practical (threshold-aware) relations ----------
def practical_eq(a: ObjVec, b: ObjVec, F_H: ObjVec) -> bool:
    """Eq. 13 line 1: per-objective equality OR both below the corresponding threshold."""
    return all(
        (a[i] == b[i]) or (a[i] <= F_H[i] and b[i] <= F_H[i]) for i in range(len(a))
    )


def practical_lt(a: ObjVec, b: ObjVec, F_H: ObjVec) -> bool:
    """Eq. 13 line 2: practical strict-less-than."""
    for i in range(len(a)):
        if a[i] < b[i] and b[i] > F_H[i]:
            # check all earlier objectives are practically-equal
            if all(
                (a[j] == b[j]) or (a[j] <= F_H[j] and b[j] <= F_H[j]) for j in range(i)
            ):
                return True
        # if not strictly less here, but earlier objectives differ, stop
        if a[i] != b[i] and not (a[i] <= F_H[i] and b[i] <= F_H[i]):
            return False
    return False


def practical_le(a: ObjVec, b: ObjVec, F_H: ObjVec) -> bool:
    return practical_lt(a, b, F_H) or practical_eq(a, b, F_H)


def is_better(F_new: ObjVec, F_cur: ObjVec, F_H: ObjVec) -> bool:
    """The 'update' predicate from Algorithm 2:
    F_new <_(F_H) F_cur,  OR  (F_new ==_(F_H) F_cur AND F_new < F_cur).
    """
    if practical_lt(F_new, F_cur, F_H):
        return True
    if practical_eq(F_new, F_cur, F_H) and lex_lt(F_new, F_cur):
        return True
    return False
