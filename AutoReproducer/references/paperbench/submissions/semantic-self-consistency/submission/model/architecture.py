"""
Core model classes for Semantic Self-Consistency replication.

Implements:
  - `Generator` -- abstract base, with concrete `OpenAIGenerator` (for
    gpt-3.5-turbo / gpt-4o-mini per addendum.md) and `HFGenerator` (for
    open-weight models such as Mistral 7B, Llama-2/3 -- not required by
    the addendum but included for completeness vs. paper §3.1).
  - `Featurizer` -- BERT-family encoder producing a single mean-pooled
    embedding per rationale (paper §3.2 / §4 step 2: "we obtain a single
    vector representation for each entire reasoning path").
  - `SemanticConsistency` -- the voter implementing both methods of paper
    §4.1 (CPW, SCW) and §4.2 (isolation forest, kNN, one-class SVM).

Reference -- paper extends:
  Wang et al. 2022, "Self-Consistency Improves Chain of Thought Reasoning
  in Language Models," arXiv:2203.11171
  (DBLP: journals/corr/abs-2203-11171; arXiv DOI not in CrossRef but
  metadata verified via DBLP record).
"""

from __future__ import annotations

import os
import math
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# ---- Optional heavy deps imported lazily ----
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    AutoModel = None  # type: ignore

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import NearestNeighbors
    from sklearn.svm import OneClassSVM
except ImportError:  # pragma: no cover
    IsolationForest = None  # type: ignore
    NearestNeighbors = None  # type: ignore
    OneClassSVM = None  # type: ignore


# ============================================================================
# Generators -- produce k chain-of-thought rationales per query (paper §4 step 1)
# ============================================================================


class Generator:
    """Base class for any rationale generator (abstract)."""

    def generate(
        self,
        prompt: str,
        k: int = 10,
        temperature: float = 0.8,
        max_new_tokens: int = 250,
        system_prompt: str | None = None,
    ) -> list[str]:
        """Return `k` sampled completions for `prompt`."""
        raise NotImplementedError


class OpenAIGenerator(Generator):
    """OpenAI chat-completion generator for gpt-3.5-turbo / gpt-4o-mini.

    Per addendum.md (lines 13-15): a system prompt of
        "Follow the exact formatting as demonstrated in the examples."
    is used, and "all examples are passed in one message, rather than in a
    chat-based conversation."
    """

    DEFAULT_SYSTEM = "Follow the exact formatting as demonstrated in the examples."

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._client = None  # lazy

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # local import for static analysis

            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def generate(
        self,
        prompt: str,
        k: int = 10,
        temperature: float = 0.8,
        max_new_tokens: int = 250,
        system_prompt: str | None = None,
    ) -> list[str]:
        client = self._get_client()
        sys_msg = system_prompt or self.DEFAULT_SYSTEM
        messages = [
            {"role": "system", "content": sys_msg},
            # addendum: examples passed in one message
            {"role": "user", "content": prompt},
        ]
        # Some OpenAI endpoints support `n` for parallel sampling;
        # we use it when available, otherwise loop.
        outputs: list[str] = []
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                n=k,
                temperature=temperature,
                max_tokens=max_new_tokens,
                top_p=0.95,
            )
            outputs = [c.message.content or "" for c in resp.choices]
        except Exception:
            # Fall back to k separate calls
            for _ in range(k):
                try:
                    resp = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_new_tokens,
                        top_p=0.95,
                    )
                    outputs.append(resp.choices[0].message.content or "")
                except Exception as e:  # pragma: no cover
                    outputs.append("")
                    time.sleep(1.0)
        return outputs


class HFGenerator(Generator):
    """Hugging Face text-generation pipeline for open-weight models.

    Not required for the replication (addendum line 1) but kept to faithfully
    mirror paper §3.1.  Uses `top_k=50, top_p=0.95, do_sample=True` per
    Appendix I.3.
    """

    def __init__(self, model_name: str, device: str | None = None):
        if AutoTokenizer is None:
            raise RuntimeError("transformers/torch not installed")
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad() if torch is not None else (lambda f: f)
    def generate(
        self,
        prompt: str,
        k: int = 10,
        temperature: float = 0.8,
        max_new_tokens: int = 250,
        system_prompt: str | None = None,
    ) -> list[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            num_return_sequences=k,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gens = self.tokenizer.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return list(gens)


# ============================================================================
# Featurizer -- BERT-family encoder, paper §3.2 / §4 step 2
# ============================================================================


class Featurizer:
    """Mean-pooled BERT-style embedding for an entire rationale.

    Paper §4 step 2:  "Instead of focusing on individual sentences or tokens,
    we obtain a single vector representation for each entire reasoning path,
    capturing its overall semantic content."

    Per §3.2 + addendum line 9:
      - StrategyQA      -> roberta-base (125M params)
      - AQuA-RAT, SVAMP -> allenai/scibert_scivocab_uncased
    """

    def __init__(
        self, model_name: str, device: str | None = None, max_length: int = 512
    ):
        if AutoTokenizer is None:
            raise RuntimeError("transformers/torch not installed")
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = max_length

    def _encode(self, texts: Sequence[str]) -> "np.ndarray":
        if not texts:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        with torch.no_grad():
            enc = self.tokenizer(
                list(texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            out = self.model(**enc)
            # Mean-pool over non-pad tokens
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            emb = summed / denom
            return emb.cpu().float().numpy()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch of rationales -> (N, d) numpy array."""
        return self._encode(texts)


# ============================================================================
# Semantic-weighting voter -- paper §4.1 (CPW, SCW) and §4.2 (outlier removal)
# ============================================================================


@dataclass
class VoteResult:
    """Container for a method's prediction on a single example."""

    method: str
    answer: str | None
    weights: dict[str, float] = field(default_factory=dict)
    kept_indices: list[int] = field(default_factory=list)


class SemanticConsistency:
    """Implements every aggregation method in the paper.

    Conventions:
      - `embeddings` is a (N, d) array where N = num generated rationales.
      - `answers` is a list of N parsed final answers (strings) -- entries
        may be None when an answer could not be parsed; these are dropped
        before voting (per addendum line 7).
    """

    # ----------------- Baselines ------------------------------------------------

    @staticmethod
    def top_prob(answers: Sequence[str | None]) -> VoteResult:
        """Single-sample baseline: take the first generation's parsed answer.
        (`top prob sample` in Table 1; addendum says generations without a
         parseable answer are excluded from the metric.)
        """
        for i, a in enumerate(answers):
            if a is not None:
                return VoteResult("top_prob", a, kept_indices=[i])
        return VoteResult("top_prob", None)

    @staticmethod
    def sc_baseline(answers: Sequence[str | None]) -> VoteResult:
        """Wang et al. 2022 self-consistency: majority vote over k samples."""
        valid = [(i, a) for i, a in enumerate(answers) if a is not None]
        if not valid:
            return VoteResult("sc_baseline", None)
        cnt = Counter(a for _, a in valid)
        winner, _ = cnt.most_common(1)[0]
        kept = [i for i, a in valid if a == winner]
        weights = {a: float(c) for a, c in cnt.items()}
        return VoteResult("sc_baseline", winner, weights=weights, kept_indices=kept)

    # ----------------- §4.1.1  Centroid Proximity Weighting --------------------

    @staticmethod
    def cpw(
        embeddings: np.ndarray, answers: Sequence[str | None], eps: float = 1e-12
    ) -> VoteResult:
        """Centroid Proximity Weighting (paper §4.1.1).

        centroid     = (1/N) * Σ_i e_i
        d_i          = || e_i - centroid ||
        d̂_i         = d_i / Σ_j d_j                (normalize)
        w_i          = 1 / d̂_i                      (inverse-proportional)
        score(u)     = Σ_{i in I(u)} w_i
        prediction   = argmax_u score(u)
        """
        valid_idx = [i for i, a in enumerate(answers) if a is not None]
        if not valid_idx:
            return VoteResult("cpw", None)
        emb = embeddings[valid_idx]
        ans = [answers[i] for i in valid_idx]

        centroid = emb.mean(axis=0)
        diffs = emb - centroid
        distances = np.linalg.norm(diffs, axis=1)  # d_i
        total = distances.sum() + eps
        normalized = distances / total  # d̂_i
        weights = 1.0 / (normalized + eps)  # w_i

        scores: dict[str, float] = defaultdict(float)
        kept_per_answer: dict[str, list[int]] = defaultdict(list)
        for w, a, gi in zip(weights, ans, valid_idx):
            scores[a] += float(w)
            kept_per_answer[a].append(gi)

        winner = max(scores, key=scores.get)
        return VoteResult(
            "cpw", winner, weights=dict(scores), kept_indices=kept_per_answer[winner]
        )

    # ----------------- §4.1.2  Semantic Consensus Weighting --------------------

    @staticmethod
    def scw(
        embeddings: np.ndarray, answers: Sequence[str | None], eps: float = 1e-12
    ) -> VoteResult:
        """Semantic Consensus Weighting via cosine similarity (paper §4.1.2).

        S_i = Σ_{j} cos(e_i, e_j)
        score(u) = Σ_{i in I(u)} S_i
        prediction = argmax_u score(u)
        """
        valid_idx = [i for i, a in enumerate(answers) if a is not None]
        if not valid_idx:
            return VoteResult("scw", None)
        emb = embeddings[valid_idx]
        ans = [answers[i] for i in valid_idx]

        # Cosine similarity matrix
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + eps
        normed = emb / norms
        sim = normed @ normed.T  # (N, N)
        per_sample = sim.sum(axis=1)  # S_i

        scores: dict[str, float] = defaultdict(float)
        kept_per_answer: dict[str, list[int]] = defaultdict(list)
        for s, a, gi in zip(per_sample, ans, valid_idx):
            scores[a] += float(s)
            kept_per_answer[a].append(gi)

        winner = max(scores, key=scores.get)
        return VoteResult(
            "scw", winner, weights=dict(scores), kept_indices=kept_per_answer[winner]
        )

    # ----------------- §4.2  Outlier-removal methods ---------------------------

    @staticmethod
    def _vote_after_filter(
        answers: Sequence[str | None], keep_mask: np.ndarray, method: str
    ) -> VoteResult:
        """Majority vote across rationales kept by an outlier filter."""
        kept = [
            i for i, (a, m) in enumerate(zip(answers, keep_mask)) if m and a is not None
        ]
        if not kept:
            # Filter removed everything -> fall back to plain SC majority
            fallback = SemanticConsistency.sc_baseline(answers)
            return VoteResult(
                method=method,
                answer=fallback.answer,
                weights=fallback.weights,
                kept_indices=fallback.kept_indices,
            )
        cnt = Counter(answers[i] for i in kept)
        winner, _ = cnt.most_common(1)[0]
        weights = {a: float(c) for a, c in cnt.items()}
        kept_winner = [i for i in kept if answers[i] == winner]
        return VoteResult(method, winner, weights=weights, kept_indices=kept_winner)

    @staticmethod
    def isolation_forest(
        embeddings: np.ndarray,
        answers: Sequence[str | None],
        n_estimators: int = 200,
        contamination="auto",
        max_samples="auto",
        random_state: int = 42,
    ) -> VoteResult:
        """Isolation Forest outlier removal (Liu et al. 2008; paper §4.2 + I.2.2).

        Anomaly score per Liu et al.:  s(x, n) = 2^(-E(h(x))/c(n))
        We use sklearn's `IsolationForest.predict` which returns +1 (inlier)
        / -1 (outlier).
        """
        if IsolationForest is None:
            raise RuntimeError("scikit-learn not installed")
        if len(embeddings) == 0:
            return VoteResult("isolation_forest", None)
        clf = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
        )
        try:
            clf.fit(embeddings)
            keep = clf.predict(embeddings) == 1
        except ValueError:
            keep = np.ones(len(embeddings), dtype=bool)
        return SemanticConsistency._vote_after_filter(answers, keep, "isolation_forest")

    @staticmethod
    def knn_outlier(
        embeddings: np.ndarray,
        answers: Sequence[str | None],
        n_neighbors: int = 5,
        metric: str = "euclidean",
        algorithm: str = "ball_tree",
        threshold_pct: float = 90.0,
    ) -> VoteResult:
        """k-Nearest-Neighbor outlier removal (paper §4.2 + I.2.1).

        Distance:  sqrt(Σ_i (x_i - y_i)^2)   (Euclidean)
        Filter:   keep points whose mean kNN distance is below the
                  `threshold_pct` percentile of all such distances.
        """
        if NearestNeighbors is None:
            raise RuntimeError("scikit-learn not installed")
        n = len(embeddings)
        if n == 0:
            return VoteResult("knn_outlier", None)
        k = min(n_neighbors + 1, n)  # include self -> we'll drop column 0
        nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm=algorithm)
        nn.fit(embeddings)
        dists, _ = nn.kneighbors(embeddings)
        mean_d = dists[:, 1:].mean(axis=1) if k > 1 else dists.mean(axis=1)
        thr = np.percentile(mean_d, threshold_pct)
        keep = mean_d <= thr
        return SemanticConsistency._vote_after_filter(answers, keep, "knn_outlier")

    @staticmethod
    def ocsvm(
        embeddings: np.ndarray,
        answers: Sequence[str | None],
        kernel: str = "linear",
        nu: float = 0.01,
        gamma: str = "scale",
    ) -> VoteResult:
        """One-class SVM outlier removal (Manevitz & Yousef 2002; §4.2 + I.2.3).

        Objective:  min  (1/2) ω^T ω + C Σ_i ζ_i
        sklearn parametrizes this via `nu`.
        """
        if OneClassSVM is None:
            raise RuntimeError("scikit-learn not installed")
        if len(embeddings) == 0:
            return VoteResult("ocsvm", None)
        clf = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        try:
            clf.fit(embeddings)
            keep = clf.predict(embeddings) == 1
        except ValueError:
            keep = np.ones(len(embeddings), dtype=bool)
        return SemanticConsistency._vote_after_filter(answers, keep, "ocsvm")

    # ----------------- Dispatcher ---------------------------------------------

    @staticmethod
    def vote(
        method: str,
        embeddings: np.ndarray | None,
        answers: Sequence[str | None],
        **kwargs,
    ) -> VoteResult:
        """Single entry-point used by eval.py."""
        m = method.lower()
        if m == "top_prob":
            return SemanticConsistency.top_prob(answers)
        if m == "sc_baseline":
            return SemanticConsistency.sc_baseline(answers)
        if embeddings is None:
            raise ValueError(f"Method `{method}` requires embeddings.")
        if m == "cpw":
            return SemanticConsistency.cpw(embeddings, answers)
        if m == "scw":
            return SemanticConsistency.scw(embeddings, answers)
        if m == "isolation_forest":
            return SemanticConsistency.isolation_forest(embeddings, answers, **kwargs)
        if m == "knn_outlier":
            return SemanticConsistency.knn_outlier(embeddings, answers, **kwargs)
        if m == "ocsvm":
            return SemanticConsistency.ocsvm(embeddings, answers, **kwargs)
        raise ValueError(f"Unknown method: {method}")
