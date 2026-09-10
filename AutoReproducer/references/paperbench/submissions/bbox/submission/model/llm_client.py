"""
Black-box LLM client for BBox-Adapter.

The paper's method only requires three operations from the LLM:

  (a) generate_complete(prompt, n)        -> n full CoT answers (single-step
                                             inference variant, Table 4)
  (b) generate_sentence(prompt, n)        -> n one-sentence continuations
                                             (used by sentence_beam_search,
                                             §3.3 / Eq. 4)
  (c) ai_feedback_select(question, cands) -> index of the best candidate
                                             according to the criteria of
                                             Appendix G (Coherency,
                                             Reasonability, Correctness,
                                             Format).

Three backends are provided so the loop runs in any environment:

  * `OpenAIClient`  — calls the OpenAI Chat Completions API; matches the
                      paper's gpt-3.5-turbo / gpt-4 setup (Appendix H.2).
  * `HFClient`      — calls a local HuggingFace causal LM (e.g.
                      Mixtral-8x7B-v0.1, the second base model in §4.7).
  * `DummyClient`   — deterministic stub used by `reproduce.sh` when no
                      API key / GPU is available.  It produces plausible
                      sentence-level CoT continuations so the rest of the
                      pipeline can be exercised end-to-end.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional


# ----------------------------------------------------------------------
# Common interface
# ----------------------------------------------------------------------
class BaseLLMClient:
    name: str = "base"

    def generate_complete(self, prompt: str, n: int = 1) -> List[str]:
        raise NotImplementedError

    def generate_sentence(self, prompt: str, n: int = 1) -> List[str]:
        raise NotImplementedError

    def ai_feedback_select(
        self,
        question: str,
        candidates: List[str],
        prompt_template: str,
    ) -> int:
        """Return the index of the best candidate (0-based)."""
        raise NotImplementedError


# ----------------------------------------------------------------------
# OpenAI backend (gpt-3.5-turbo, gpt-4)
# ----------------------------------------------------------------------
class OpenAIClient(BaseLLMClient):
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 1.0,
        max_tokens: int = 512,
        api_key: Optional[str] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "openai package not installed; pip install openai>=1.12.0"
            ) from e
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _chat(
        self, prompt: str, n: int, stop: Optional[Iterable[str]] = None
    ) -> List[str]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=n,
            stop=list(stop) if stop else None,
        )
        return [c.message.content or "" for c in resp.choices]

    def generate_complete(self, prompt: str, n: int = 1) -> List[str]:
        return self._chat(prompt, n=n)

    def generate_sentence(self, prompt: str, n: int = 1) -> List[str]:
        # Stop at the first newline / sentence terminator so we get one
        # sentence per call — see §3.3 "decompose ... into sentence-level
        # beam search".
        return self._chat(prompt, n=n, stop=["\n", ".  "])

    def ai_feedback_select(
        self, question: str, candidates: List[str], prompt_template: str
    ) -> int:
        cand_block = "\n".join(
            f"[Candidate Answer {i + 1}]: {c}" for i, c in enumerate(candidates)
        )
        prompt = prompt_template.replace("<QUESTION>", question).replace(
            "<CANDIDATE_ANSWERS>", cand_block
        )
        text = self._chat(prompt, n=1)[0]
        # Look for the integer index in the model response.
        m = re.search(r"\[Candidate Answer\s*(\d+)\]", text)
        if not m:
            return 0
        idx = int(m.group(1)) - 1
        return max(0, min(idx, len(candidates) - 1))


# ----------------------------------------------------------------------
# HuggingFace backend — Mixtral-8x7B-v0.1 path from §4.7.
# ----------------------------------------------------------------------
class HFClient(BaseLLMClient):
    name = "hf"

    def __init__(
        self,
        model: str = "mistralai/Mixtral-8x7B-v0.1",
        temperature: float = 1.0,
        max_tokens: int = 512,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch_dtype, device_map=device
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device

    def _generate(
        self, prompt: str, n: int, stop_token: Optional[str] = None
    ) -> List[str]:
        import torch

        ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = []
        for _ in range(n):
            with torch.no_grad():
                out = self.model.generate(
                    **ids,
                    do_sample=True,
                    temperature=self.temperature,
                    max_new_tokens=self.max_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(
                out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True
            )
            if stop_token and stop_token in text:
                text = text.split(stop_token)[0]
            outputs.append(text)
        return outputs

    def generate_complete(self, prompt, n=1):
        return self._generate(prompt, n=n)

    def generate_sentence(self, prompt, n=1):
        return self._generate(prompt, n=n, stop_token="\n")

    def ai_feedback_select(self, question, candidates, prompt_template):
        cand_block = "\n".join(
            f"[Candidate Answer {i + 1}]: {c}" for i, c in enumerate(candidates)
        )
        prompt = prompt_template.replace("<QUESTION>", question).replace(
            "<CANDIDATE_ANSWERS>", cand_block
        )
        text = self._generate(prompt, n=1)[0]
        m = re.search(r"\[Candidate Answer\s*(\d+)\]", text)
        if not m:
            return 0
        idx = int(m.group(1)) - 1
        return max(0, min(idx, len(candidates) - 1))


# ----------------------------------------------------------------------
# Dummy backend — deterministic, no external dependencies.
# ----------------------------------------------------------------------
class DummyClient(BaseLLMClient):
    """A deterministic, hash-based pseudo-LLM used by the smoke run.

    For each `(prompt, n)` call we return `n` distinct one-line strings
    derived from the prompt's hash so beam search has something
    non-trivial to rank.  Ground-truth questions whose answers appear
    in the prompt are echoed verbatim — this lets the smoke run
    exercise the *outcome supervision* path of §4.1.
    """

    name = "dummy"

    def __init__(self, temperature: float = 1.0, max_tokens: int = 512, **_):
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _hash(self, s: str, salt: int) -> int:
        # Stable hash — Python's hash() is randomised across runs.
        h = 0
        for c in s:
            h = (h * 131 + ord(c)) & 0xFFFFFFFF
        return (h + salt) & 0xFFFFFFFF

    def _candidate(self, prompt: str, k: int) -> str:
        h = self._hash(prompt, k)
        templates = [
            f"Step {k + 1}: We compute a sub-result of {h % 97}.",
            f"Step {k + 1}: Combining the previous reasoning we get {h % 53}.",
            f"Step {k + 1}: After simplification the value is {h % 31}.",
            f"Step {k + 1}: It follows by direct computation that x = {h % 71}.",
        ]
        return templates[k % len(templates)]

    def generate_complete(self, prompt: str, n: int = 1) -> List[str]:
        out = []
        for k in range(n):
            steps = [self._candidate(prompt + str(s), k) for s in range(3)]
            steps.append(f"#### {self._hash(prompt, k) % 100}")
            out.append("\n".join(steps))
        return out

    def generate_sentence(self, prompt: str, n: int = 1) -> List[str]:
        return [self._candidate(prompt, k) for k in range(n)]

    def ai_feedback_select(
        self, question: str, candidates: List[str], prompt_template: str
    ) -> int:
        # Deterministic: pick the candidate whose hash is largest.
        scores = [self._hash(c, i) for i, c in enumerate(candidates)]
        return int(max(range(len(candidates)), key=lambda i: scores[i]))


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------
@dataclass
class LLMConfig:
    backend: str = "dummy"
    name: str = "gpt-3.5-turbo"
    temperature: float = 1.0
    max_tokens: int = 512


def build_llm_client(cfg: LLMConfig) -> BaseLLMClient:
    if cfg.backend == "openai":
        return OpenAIClient(
            model=cfg.name, temperature=cfg.temperature, max_tokens=cfg.max_tokens
        )
    if cfg.backend == "hf":
        return HFClient(
            model=cfg.name, temperature=cfg.temperature, max_tokens=cfg.max_tokens
        )
    if cfg.backend == "dummy":
        return DummyClient(temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    raise ValueError(f"Unknown LLM backend: {cfg.backend}")
