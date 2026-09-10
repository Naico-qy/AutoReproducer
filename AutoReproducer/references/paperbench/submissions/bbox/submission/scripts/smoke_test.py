"""
Lightweight smoke test for BBox-Adapter — exercises every component
of the codebase in a few seconds, with no external dependencies
beyond `torch`, `transformers`, and `pyyaml`.

Run from the submission root:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import sys

# Make sibling packages importable when running from `scripts/`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    import torch
    from data.synthetic import build_synthetic_strategyqa
    from data.pools import select_positive
    from model.architecture import AdapterConfig, BBoxAdapter
    from model.llm_client import LLMConfig, build_llm_client
    from model.loss import compute_nce_batch_loss
    from model.inference import sentence_beam_search

    torch.manual_seed(0)

    # ---- 1. Synthetic dataset
    examples = build_synthetic_strategyqa(n=4, seed=0)
    print(f"[smoke] {len(examples)} synthetic examples")

    # ---- 2. Adapter (small / mock backbone for speed)
    cfg = AdapterConfig(backbone="prajjwal1/bert-tiny", hidden_dim=128, max_length=128)
    try:
        adapter = BBoxAdapter(cfg)
    except Exception as e:
        print(f"[smoke] (skipping HF download in this env: {e})")
        return
    print(f"[smoke] adapter params = {adapter.num_parameters / 1e6:.2f}M")

    # ---- 3. LLM client (dummy)
    llm = build_llm_client(LLMConfig(backend="dummy"))

    # ---- 4. SEL + pools
    for ex in examples:
        cands = llm.generate_complete(f"Q: {ex.question}\nA:", n=3)
        pos, negs = select_positive(
            mode="ground_truth",
            question=ex.question,
            gold_answer=ex.answer,
            prev_positive=ex.rationale,
            candidates=cands,
            llm_client=llm,
            ai_feedback_prompt="",
        )
        print(f"[smoke] qid={ex.qid}  pos={pos[:30]!r}  #neg={len(negs)}")

    # ---- 5. Loss
    qs = [ex.question for ex in examples]
    pos = [ex.rationale for ex in examples]
    negs = [llm.generate_complete(ex.question, n=3) for ex in examples]
    stats = compute_nce_batch_loss(adapter, qs, pos, negs, alpha=0.01)
    print(f"[smoke] loss={stats['loss'].item():.4f}  acc@1={stats['acc@1'].item():.3f}")

    # ---- 6. Beam search
    out = sentence_beam_search(
        adapter,
        llm,
        examples[0].question,
        prompt_template="Q: <QUESTION>\nA: Let's think step by step.\n",
        beam_size=2,
        num_candidates_per_step=2,
        max_steps=3,
        stop_token="####",
    )
    print(f"[smoke] beam-search out (first 60 chars) = {out[:60]!r}")

    print("[smoke] OK")


if __name__ == "__main__":
    main()
