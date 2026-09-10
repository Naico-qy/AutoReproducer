"""
Answer-parsing utilities.

Per addendum.md (lines 3-7):
    "the answer is parsed as the string immediately following the string
    'the answer is ' (being case insensitive), removing any whitespace,
    fullstops, and parentheses."

    e.g.  "The answer is 2."   -> "2"
          "The answer is (B)"  -> "B"

For top-prob (and AQuA-RAT for *all* methods), generations are only
considered if an answer can be extracted (addendum line 7).
"""

from __future__ import annotations

import re
import string

# Precompiled: split on first occurrence of "the answer is" (case-insensitive)
_ANSWER_RE = re.compile(r"the answer is", re.IGNORECASE)
_STRIP_CHARS = string.whitespace + ".()[]{}<>"


def extract_answer(text: str) -> str | None:
    """Return the substring after the FIRST 'the answer is', stripped of
    whitespace, fullstops, and parentheses.  Returns None if not found.
    """
    if text is None:
        return None
    m = _ANSWER_RE.search(text)
    if m is None:
        return None
    tail = text[m.end() :]
    # Take everything up to a newline (so multi-question generations don't
    # leak the next 'Q:' into the parsed answer)
    tail = tail.splitlines()[0] if "\n" in tail else tail
    # Strip outer whitespace/fullstops/parens
    tail = tail.strip(_STRIP_CHARS)
    # The addendum strips *all* whitespace/fullstops/parens (line 4),
    # not just from the ends. We follow that literally:
    cleaned = "".join(c for c in tail if c not in "().[]{}<> \t\r\n.,!")
    # But preserve internal hyphens/numbers/word chars
    return cleaned if cleaned else None


def normalize_answer(ans: str | None, kind: str = "auto") -> str | None:
    """Normalize a parsed answer for comparison.

    kind:
      "letter"  -> lowercase single letter (AQuA-RAT (a)-(e))
      "number"  -> string of digits / signed float (SVAMP)
      "boolean" -> 'yes' or 'no' (StrategyQA)
      "auto"    -> heuristic: tries letter, then number, then yes/no
    """
    if ans is None:
        return None
    a = ans.strip().lower()

    if kind == "letter":
        # take first alphabetic character
        for c in a:
            if c.isalpha():
                return c
        return None

    if kind == "number":
        # Pull the first signed numeric out of `a`
        m = re.search(r"-?\d+\.?\d*", a)
        if not m:
            return None
        v = m.group(0)
        # Drop trailing ".0" so "8" == "8.0"
        if "." in v:
            try:
                f = float(v)
                if float(int(f)) == f:
                    return str(int(f))
                return str(f)
            except ValueError:
                return v
        return v

    if kind == "boolean":
        if a.startswith("y") or "true" in a:
            return "yes"
        if a.startswith("n") or "false" in a:
            return "no"
        return None

    # auto fallback (used by self-consistency vote-grouping where we don't
    # know the dataset type a priori)
    if a in {"yes", "no", "true", "false"}:
        return "yes" if a in {"yes", "true"} else "no"
    if len(a) == 1 and a.isalpha():
        return a
    n = normalize_answer(a, "number")
    if n is not None:
        return n
    return a


def score_prediction(pred: str | None, gold: str, kind: str = "auto") -> int:
    """Return 1 if pred matches gold under `kind`, else 0."""
    p = normalize_answer(pred, kind)
    g = normalize_answer(gold, kind)
    if p is None or g is None:
        return 0
    return int(p == g)
