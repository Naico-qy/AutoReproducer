"""Evaluation entrypoint.

Reproduces the Section 3.3 / 5 / 6 metrics:
  * Toxicity   -- 1199 challenge prompts from RealToxicityPrompts, scored with
                  unitary/unbiased-toxic-roberta (per addendum.md).
  * Perplexity -- sliding-window PPL on Wikitext-2 (test split).
  * F1         -- harmonic mean of token precision/recall vs the original
                  Wikipedia continuation, on 2,000 Wikipedia prompts.

Also reproduces:
  * Table 2 / 7 -- intervention rows (SUBTRACT W_toxic / MLP.v / SVD.U).
  * Table 4     -- un-aligning by scaling the top-7 MLP.k_toxic by 10x.

Outputs JSON to ``cfg["output"]["metrics_json"]`` (default ``./outputs/metrics.json``)
which the PaperBench Full-mode judge reads from ``/output/metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from data import (
    iter_wikitext_chunks,
    load_real_toxicity_prompts,
    load_wikitext_prompts,
)
from interventions import (
    apply_residual_subtraction,
    apply_un_align_key_scaling,
)
from model.architecture import GPT2WithHooks


# ---------------------------------------------------------------------------
# Toxicity scoring (per addendum: use unitary/unbiased-toxic-roberta)
# ---------------------------------------------------------------------------
class ToxicityScorer:
    def __init__(self, model_name: str, device: str):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_name)
            .to(device)
            .eval()
        )
        self.device = device

    @torch.no_grad()
    def score_batch(self, texts: list[str], batch_size: int = 16) -> list[float]:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tok(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            # unitary/unbiased-toxic-roberta has 16 labels; index 0 = "toxicity"
            probs = torch.sigmoid(logits)
            tox = probs[:, 0].cpu().tolist()
            out.extend(tox)
        return out


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_continuations(
    model: GPT2WithHooks,
    prompts: list[str],
    gen_len: int,
    device: str,
    batch_size: int = 8,
) -> list[str]:
    outs: list[str] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="gen"):
        batch = prompts[i : i + batch_size]
        enc = model.tokenizer(
            batch, padding=True, truncation=True, max_length=64, return_tensors="pt"
        ).to(device)
        gen = model.lm.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=gen_len,
            do_sample=False,
            pad_token_id=model.tokenizer.pad_token_id,
        )
        for j, ids in enumerate(gen):
            cont = ids[enc.input_ids.shape[1] :]
            outs.append(model.tokenizer.decode(cont, skip_special_tokens=True))
    return outs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_perplexity(
    model: GPT2WithHooks, seq_len: int = 1024, device: str = "cuda"
) -> float:
    """Sliding-window perplexity on Wikitext-2 test (Section 3.3 perplexity)."""
    nlls = []
    total = 0
    for chunk in iter_wikitext_chunks(seq_len=seq_len * 4, split="test"):
        ids = model.tokenizer(
            chunk, return_tensors="pt", truncation=True, max_length=seq_len
        ).input_ids.to(device)
        if ids.shape[1] < 2:
            continue
        out = model.lm(ids, labels=ids)
        nlls.append(float(out.loss.item()) * (ids.shape[1] - 1))
        total += ids.shape[1] - 1
    return math.exp(sum(nlls) / max(1, total))


def compute_f1(
    model: GPT2WithHooks, n_prompts: int, gen_len: int, device: str
) -> float:
    """Section 3.3 F1: harmonic mean of token-level precision and recall vs the
    original Wikipedia continuation.  Uses 2000 Wikipedia sentences as prompts.
    """
    # Build (prompt, gold continuation) pairs by splitting wikitext sentences in half.
    sentences = load_wikitext_prompts(
        n_prompts=n_prompts * 2, split="train", min_len=24
    )
    pairs: list[tuple[str, str]] = []
    for s in sentences:
        toks = s.split()
        if len(toks) < 8:
            continue
        mid = len(toks) // 2
        pairs.append((" ".join(toks[:mid]), " ".join(toks[mid:])))
        if len(pairs) >= n_prompts:
            break

    prompts = [p for p, _ in pairs]
    golds = [g for _, g in pairs]
    gens = generate_continuations(model, prompts, gen_len=gen_len, device=device)

    f1s = []
    for gen, gold in zip(gens, golds):
        gen_tok = set(gen.split())
        gold_tok = set(gold.split())
        if not gen_tok or not gold_tok:
            continue
        inter = gen_tok & gold_tok
        if not inter:
            f1s.append(0.0)
            continue
        prec = len(inter) / len(gen_tok)
        rec = len(inter) / len(gold_tok)
        f1s.append(2 * prec * rec / (prec + rec))
    return sum(f1s) / max(1, len(f1s))


def evaluate_model(
    model: GPT2WithHooks,
    cfg: dict,
    scorer: ToxicityScorer,
    device: str,
    label: str,
) -> dict:
    """Compute the three columns of Table 2 (Toxic / PPL / F1)."""
    print(f"\n=== Evaluating: {label} ===")
    rtp_prompts = load_real_toxicity_prompts(
        max_prompts=cfg["eval"]["toxicity_prompts"]["max_prompts"],
        split=cfg["eval"]["toxicity_prompts"]["split"],
    )
    gens = generate_continuations(
        model,
        rtp_prompts,
        gen_len=cfg["eval"]["toxicity_prompts"]["gen_length"],
        device=device,
    )
    tox_scores = scorer.score_batch(gens)
    toxicity = sum(tox_scores) / max(1, len(tox_scores))

    ppl = compute_perplexity(
        model, seq_len=cfg["eval"]["perplexity"]["seq_len"], device=device
    )
    f1 = compute_f1(
        model,
        n_prompts=cfg["eval"]["f1"]["n_prompts"],
        gen_len=cfg["eval"]["f1"]["gen_length"],
        device=device,
    )

    metrics = {"label": label, "toxicity": toxicity, "perplexity": ppl, "f1": f1}
    print(metrics)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--quick", action="store_true", help="Use small subsets for smoke tests."
    )
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.quick:
        cfg["eval"]["toxicity_prompts"]["max_prompts"] = 32
        cfg["eval"]["f1"]["n_prompts"] = 16
        cfg["eval"]["perplexity"]["seq_len"] = 256

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.get("seed", 42))

    scorer = ToxicityScorer(cfg["eval"]["toxicity_model"], device)

    # ---- Pre-DPO baseline (Table 2 row "NO OP") ----
    base_model = GPT2WithHooks(cfg["model"]["name"]).to(device)
    base_model.eval()
    metrics_base = evaluate_model(base_model, cfg, scorer, device, label="GPT2 (NO OP)")

    # ---- Load toxic vectors (W_toxic, MLP.v_toxic, SVD.U_toxic) ----
    tox_path = Path(cfg["output"]["toxic_vectors"])
    intervention_metrics = []
    if tox_path.exists():
        tox = torch.load(tox_path, map_location="cpu")
        W_toxic = tox["toxic_direction"]  # [d_model]
        v_top = tox["V_toxic"][0]  # most toxic MLP.v
        U0 = tox["U_toxic"][:, 0]  # SVD.U_toxic[0]

        for name, vec, alpha_key in [
            ("SUBTRACT W_toxic", W_toxic, "w_toxic"),
            ("SUBTRACT MLP.v_top", v_top, "mlp_v_770_19"),
            ("SUBTRACT SVD.U_toxic[0]", U0, "svd_u_0"),
        ]:
            alpha = cfg["interventions"]["alphas"][alpha_key]
            apply_residual_subtraction(base_model, vec, alpha)
            m = evaluate_model(base_model, cfg, scorer, device, label=name)
            base_model.remove_interventions()
            intervention_metrics.append(m)
    else:
        print(
            f"WARNING: {tox_path} not found, skipping intervention eval (Table 2 rows)."
        )

    # ---- Post-DPO model (Table 2 last row) ----
    dpo_dir = Path(cfg["output"]["dpo_ckpt"])
    metrics_dpo = None
    metrics_unalign = None
    if dpo_dir.exists():
        dpo_model = GPT2WithHooks(cfg["model"]["name"]).to(device)
        dpo_model.lm.from_pretrained(str(dpo_dir)).to(device)
        dpo_model.eval()
        metrics_dpo = evaluate_model(dpo_model, cfg, scorer, device, label="GPT2_DPO")

        # ---- Section 6: un-align by scaling top-7 MLP.k_toxic by 10x ----
        if tox_path.exists():
            tox = torch.load(tox_path, map_location="cpu")
            apply_un_align_key_scaling(
                dpo_model,
                toxic_indices=tox["indices"],
                scale=cfg["unalign"]["scale_factor"],
                n_top=cfg["unalign"]["n_top_keys"],
            )
            metrics_unalign = evaluate_model(
                dpo_model, cfg, scorer, device, label="SCALE MLP.k_Toxic (un-align)"
            )
            dpo_model.remove_interventions()
    else:
        print(f"WARNING: {dpo_dir} not found, skipping post-DPO eval.")

    # ---- Save metrics ----
    out = {
        "table2": {
            "no_op": metrics_base,
            "interventions": intervention_metrics,
            "dpo": metrics_dpo,
        },
        "table4_unalign": metrics_unalign,
    }
    out_path = Path(cfg["output"]["metrics_json"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    # Also write to /output/metrics.json (PaperBench convention) if writable
    try:
        Path("/output").mkdir(parents=True, exist_ok=True)
        with open("/output/metrics.json", "w") as f:
            json.dump(out, f, indent=2)
    except Exception:  # noqa: BLE001
        pass
    print(f"\nWrote metrics -> {out_path}")


if __name__ == "__main__":
    main()
