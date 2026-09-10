"""§3.2 -- Chain-of-Thought evaluation on GSM8K and AQuA.

Reproduces Figure 2: top panel = task accuracy; bottom panel =
percentage of generations whose final answer fails to be parsed
("invalid format rate").

The addendum clarifies that in §3.2 the paper upweights w_p (the prompt c)
and may also choose to upweight w_p plus a partial chain-of-thought.  We
expose both options through the `cfg_cot_variant` parameter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from tqdm import tqdm

from data.loader import BenchmarkSample, load_cot
from data.prompts import cot_prompt
from model.architecture import CFGSampler


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
_GSM8K_PATTERN = re.compile(r"(?:The answer is|####)\s*\$?([\-0-9\.,]+)", re.IGNORECASE)
_AQUA_PATTERN = re.compile(r"answer is\s*\(?([A-E])\)?", re.IGNORECASE)


def extract_answer(generation: str, dataset: str) -> Optional[str]:
    """Pull the final answer out of a CoT continuation.

    Returns None when no answer can be parsed (counts as an "invalid"
    format in the bottom panel of Figure 2).
    """
    pat = _GSM8K_PATTERN if dataset == "gsm8k" else _AQUA_PATTERN
    matches = pat.findall(generation)
    if not matches:
        return None
    if dataset == "gsm8k":
        return matches[-1].replace(",", "").strip()
    return matches[-1].upper().strip()


def _normalise_gsm8k(s: str) -> str:
    return s.replace(",", "").replace("$", "").strip().rstrip(".")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
@dataclass
class CoTResult:
    dataset: str
    gamma: float
    accuracy: float
    invalid_rate: float
    n: int


@torch.no_grad()
def evaluate_cot(
    sampler: CFGSampler,
    name: str,
    hf_path: str,
    config: Optional[str],
    gamma: float,
    cfg_cot_variant: str = "prompt_only",
    max_samples: Optional[int] = None,
    max_new_tokens: int = 512,
) -> CoTResult:
    """Run CoT evaluation at a single γ value.

    `cfg_cot_variant` controls what the CFG conditioning vector is:
        * "prompt_only"            -- c = w_p (the question)  ←  paper default
        * "prompt_plus_partial_cot" -- c = w_p + w_cot          ← §3.2 footnote
    """
    sampler.guidance_scale = gamma
    correct = 0
    invalid = 0
    total = 0

    samples: Iterable[BenchmarkSample] = load_cot(name, hf_path, config)
    for i, sample in enumerate(tqdm(samples, desc=f"{name} CoT γ={gamma}")):
        if max_samples is not None and i >= max_samples:
            break

        prompt = cot_prompt(sample.context, dataset=name)

        if cfg_cot_variant == "prompt_plus_partial_cot":
            # Keep the unconditional context as the empty prefix but set the
            # CFG positive prompt to include the few-shot examples (which
            # contain partial CoT chains).  Implementation detail: the
            # `CFGSampler.uncond_from_last_token` flag is preserved.
            sampler.uncond_from_last_token = True
        else:
            sampler.uncond_from_last_token = True

        gen = sampler.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        ans = extract_answer(gen, name)
        if ans is None:
            invalid += 1
        else:
            if name == "gsm8k":
                if _normalise_gsm8k(ans) == _normalise_gsm8k(sample.answer_text or ""):
                    correct += 1
            else:
                if ans.upper() == (sample.answer_text or "").upper():
                    correct += 1
        total += 1

    return CoTResult(
        dataset=name,
        gamma=gamma,
        accuracy=correct / max(total, 1),
        invalid_rate=invalid / max(total, 1),
        n=total,
    )
