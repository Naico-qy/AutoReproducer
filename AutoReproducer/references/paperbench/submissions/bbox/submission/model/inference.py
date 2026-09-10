"""
Adapted inference (§3.3) for BBox-Adapter.

Two variants of inference are implemented:

  * `single_step_inference`  — Table 4's "single step" baseline: the LLM
    generates `M` complete answers in one shot and the adapter picks
    the highest-scoring one.
  * `sentence_beam_search`   — full §3.3 / Eq. (4) sentence-level beam
    search.  At each step we extend each of the `k` current beams with
    `n` LLM proposals (n*k total candidates), keep the top-`k`
    according to g_theta, and continue until either every beam emits
    the stop signal "####" or `L` steps elapse.

Reference: Wei et al., "Chain-of-Thought Prompting Elicits Reasoning",
NeurIPS 2022 — verified via `ref_verify`/CrossRef during preparation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class Beam:
    """A single beam in the sentence-level beam search."""

    steps: List[str] = field(default_factory=list)
    score: float = 0.0
    finished: bool = False

    @property
    def text(self) -> str:
        return "\n".join(self.steps)


# ----------------------------------------------------------------------
# Single-step inference (Table 4 "single step" variant)
# ----------------------------------------------------------------------
@torch.no_grad()
def single_step_inference(
    adapter,
    llm_client,
    question: str,
    prompt_template: str,
    num_candidates: int = 3,
    device: Optional[torch.device] = None,
) -> str:
    """
    Generate `num_candidates` complete answers from the LLM in one call,
    and return the one whose energy g_theta(x, y) is highest.
    """
    prompt = prompt_template.replace("<QUESTION>", question)
    candidates = llm_client.generate_complete(prompt, n=num_candidates)
    energies = adapter.score([question] * len(candidates), candidates, device=device)
    best = int(torch.argmax(energies).item())
    return candidates[best]


# ----------------------------------------------------------------------
# Full sentence-level beam search (§3.3, Eq. 4)
# ----------------------------------------------------------------------
@torch.no_grad()
def sentence_beam_search(
    adapter,
    llm_client,
    question: str,
    prompt_template: str,
    beam_size: int = 3,
    num_candidates_per_step: int = 3,
    max_steps: int = 8,
    stop_token: str = "####",
    device: Optional[torch.device] = None,
) -> str:
    """
    Sentence-level beam search exactly as in §3.3.

    Pseudocode (paper):

        For step l = 1..L:
            For each of k beams:
                sample n proposals s_l ~ p_LLM(. | x, s_{1:l-1})
            Form n*k candidate chains s_{1:l} -> candidate set C
            Score each chain with g_theta and keep top-k.
        Return the highest-scoring fully-finished chain.

    Returns the textual concatenation of the winning beam's sentences.
    """
    base_prompt = prompt_template.replace("<QUESTION>", question)

    # Initialise with a single empty beam.
    beams: List[Beam] = [Beam()]

    for _step in range(max_steps):
        # Stop early if every beam has emitted the stop signal.
        if all(b.finished for b in beams):
            break

        candidates: List[Beam] = []
        for beam in beams:
            if beam.finished:
                # Carry it through unchanged so it stays in contention.
                candidates.append(beam)
                continue

            # Build the LLM prompt = base prompt + accumulated reasoning.
            prefix = base_prompt
            if beam.steps:
                prefix = prefix + "\n" + "\n".join(beam.steps)

            proposals = llm_client.generate_sentence(prefix, n=num_candidates_per_step)
            for prop in proposals:
                prop = prop.strip()
                if not prop:
                    continue
                new_beam = Beam(
                    steps=list(beam.steps) + [prop],
                    score=beam.score,
                    finished=stop_token in prop,
                )
                candidates.append(new_beam)

        if not candidates:
            break

        # Score every candidate chain with g_theta and keep top-k.
        chains = [c.text for c in candidates]
        energies = adapter.score([question] * len(chains), chains, device=device)
        for i, c in enumerate(candidates):
            c.score = float(energies[i].item())

        candidates.sort(key=lambda b: b.score, reverse=True)
        beams = candidates[:beam_size]

    # Final selection: highest-scoring beam (paper: "the adapted
    # generation is selected based on the highest-scoring option
    # evaluated by the adapter").
    best = max(beams, key=lambda b: b.score)
    return best.text
