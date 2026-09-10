"""Main evaluation entrypoint.

Examples
--------
    python eval.py --config configs/zero_shot.yaml         --task zero_shot
    python eval.py --config configs/chain_of_thought.yaml  --task cot
    python eval.py --config configs/humaneval.yaml         --task humaneval
    python eval.py --config configs/flops.yaml             --task flops
    python eval.py --config configs/entropy.yaml           --task entropy
    python eval.py --config configs/entropy.yaml           --task ppl_corr
    python eval.py --config configs/entropy.yaml           --task visualize

The script dispatches to one of the per-section drivers depending on the
`--task` flag.  Heavy lifting (model loading, dataset iteration, CFG
sampling) lives in the corresponding `evaluation/` or `analysis/` module.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml


# ---------------------------------------------------------------------------
# Config loading with simple `inherits:` support
# ---------------------------------------------------------------------------
def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherits", None)
    if parent is not None:
        parent_path = Path(path).parent / parent
        base = _load_yaml(str(parent_path))
        # Shallow merge -- child overrides
        merged = {**base, **cfg}
        # Deep-merge `cfg.*` and `generation.*` blocks
        for k in ("cfg", "generation"):
            if k in base and k in cfg and isinstance(base[k], dict):
                merged[k] = {**base[k], **cfg[k]}
        return merged
    return cfg


def _load_model_and_tokenizer(
    model_id: str, dtype: str = "float16", device: str = "cuda"
):
    """Lazy import so that --task=flops can run without torch/transformers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Task dispatchers
# ---------------------------------------------------------------------------
def _run_zero_shot(cfg):
    from evaluation.lm_eval_harness import evaluate_zero_shot
    from model.architecture import CFGSampler

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_ids = cfg.get("model_ids") or [cfg["models"]["pythia"]["sizes"][0]]
    results = []
    for mid in model_ids:
        model, tokenizer = _load_model_and_tokenizer(mid, cfg["dtype"], cfg["device"])
        sampler = CFGSampler(
            model=model,
            tokenizer=tokenizer,
            uncond_from_last_token=cfg["cfg"]["uncond_from_last_token"],
        )
        for ds in cfg["datasets"]:
            for gamma in cfg["cfg"]["gamma_grid"]:
                r = evaluate_zero_shot(
                    sampler=sampler,
                    name=ds["name"],
                    hf_path=ds["hf_path"],
                    config=ds.get("config"),
                    metric=ds["metric"],
                    gamma=gamma,
                )
                results.append({"model": mid, **r.__dict__})
                print(json.dumps(results[-1]))
    (out_dir / "zero_shot_results.json").write_text(json.dumps(results, indent=2))


def _run_cot(cfg):
    from evaluation.chain_of_thought import evaluate_cot
    from model.architecture import CFGSampler

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for mid in cfg["models"]:
        model, tokenizer = _load_model_and_tokenizer(mid, cfg["dtype"], cfg["device"])
        sampler = CFGSampler(
            model=model,
            tokenizer=tokenizer,
            uncond_from_last_token=cfg["cfg"]["uncond_from_last_token"],
        )
        for ds in cfg["datasets"]:
            for gamma in cfg["cfg"]["gamma_grid"]:
                for variant in cfg.get(
                    "cfg_cot_variants", [{"upweight": "prompt_only"}]
                ):
                    r = evaluate_cot(
                        sampler=sampler,
                        name=ds["name"],
                        hf_path=ds["hf_path"],
                        config=ds.get("config"),
                        gamma=gamma,
                        cfg_cot_variant=variant["upweight"],
                        max_new_tokens=cfg["generation"]["max_new_tokens"],
                    )
                    results.append(
                        {"model": mid, "variant": variant["upweight"], **r.__dict__}
                    )
                    print(json.dumps(results[-1]))
    (out_dir / "cot_results.json").write_text(json.dumps(results, indent=2))


def _run_humaneval(cfg):
    from evaluation.humaneval import evaluate_humaneval
    from model.architecture import CFGSampler

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for mid in cfg["models"]:
        model, tokenizer = _load_model_and_tokenizer(mid, cfg["dtype"], cfg["device"])
        sampler = CFGSampler(
            model=model,
            tokenizer=tokenizer,
            uncond_from_last_token=cfg["cfg"]["uncond_from_last_token"],
        )
        for T in cfg["temperatures"]:
            for gamma in cfg["cfg"]["gamma_grid"]:
                r = evaluate_humaneval(
                    sampler=sampler,
                    gamma=gamma,
                    temperature=T,
                    n_samples_per_problem=cfg["n_samples_per_problem"],
                    k_values=tuple(cfg["k_values"]),
                    top_p=cfg["top_p"],
                    max_new_tokens=cfg["generation"]["max_new_tokens"],
                )
                results.append({"model": mid, **r.__dict__})
                print(json.dumps(results[-1]))
    (out_dir / "humaneval_results.json").write_text(json.dumps(results, indent=2))


def _run_flops(cfg):
    from transformers import AutoConfig
    from model.flops import count_flops_electra, hf_config_to_electra
    from evaluation.ancova import build_cost_table, run_ancova

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    flops_table = {}
    for pair in cfg["size_pairs"]:
        for k in ("small", "large"):
            mid = pair[k]
            hf = AutoConfig.from_pretrained(mid)
            ec = hf_config_to_electra(hf)
            flops_table[mid] = count_flops_electra(ec, with_cfg=False)
            flops_table[f"{mid}+CFG"] = count_flops_electra(ec, with_cfg=True)
    (out_dir / "flops.json").write_text(json.dumps(flops_table, indent=2))
    print(json.dumps(flops_table, indent=2))


def _run_entropy(cfg):
    from analysis.entropy_analysis import compare_entropy
    from data.p3_sampler import P3SamplerConfig
    from model.architecture import CFGSampler

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    base, tok = _load_model_and_tokenizer(
        cfg["base_model"], cfg["dtype"], cfg["device"]
    )
    base_sampler = CFGSampler(model=base, tokenizer=tok, guidance_scale=1.0)
    cfg_sampler = CFGSampler(
        model=base,
        tokenizer=tok,
        guidance_scale=cfg["cfg"]["gamma"],
        uncond_from_last_token=cfg["cfg"]["uncond_from_last_token"],
    )
    p3 = P3SamplerConfig(
        hf_path=cfg["dataset"]["hf_path"],
        total_samples=cfg["dataset"]["total_samples"],
        per_dataset_cap=cfg["dataset"]["per_dataset_cap"],
        max_input_tokens=cfg["dataset"]["max_input_tokens"],
        seed=cfg["dataset"]["random_seed"],
    )
    res = compare_entropy(
        base_sampler, cfg_sampler, p3, top_p=cfg["entropy"]["top_p_for_token_count"]
    )
    (out_dir / "entropy.json").write_text(json.dumps(res.__dict__, indent=2))
    print(json.dumps(res.__dict__, indent=2))


def _run_ppl_corr(cfg):
    from analysis.perplexity_corr import perplexity_correlation
    from data.p3_sampler import P3SamplerConfig

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    base, tok = _load_model_and_tokenizer(
        cfg["base_model"], cfg["dtype"], cfg["device"]
    )
    inst, _ = _load_model_and_tokenizer(
        cfg["instruct_model"], cfg["dtype"], cfg["device"]
    )
    p3 = P3SamplerConfig(
        hf_path=cfg["dataset"]["hf_path"],
        total_samples=cfg["dataset"]["total_samples"],
        per_dataset_cap=cfg["dataset"]["per_dataset_cap"],
        max_input_tokens=cfg["dataset"]["max_input_tokens"],
        seed=cfg["dataset"]["random_seed"],
    )
    res = perplexity_correlation(base, inst, tok, p3, cfg_gamma=cfg["cfg"]["gamma"])
    out = {
        "n": res.n,
        "rho_baseline_cfg": res.rho_baseline_cfg,
        "rho_baseline_instruct": res.rho_baseline_instruct,
        "rho_cfg_instruct": res.rho_cfg_instruct,
    }
    (out_dir / "ppl_corr.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def _run_visualize(cfg):
    from analysis.visualize import token_reranking_table
    from model.architecture import CFGSampler

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    base, tok = _load_model_and_tokenizer(
        cfg["base_model"], cfg["dtype"], cfg["device"]
    )
    sampler = CFGSampler(
        model=base,
        tokenizer=tok,
        guidance_scale=cfg["cfg"]["gamma"],
        uncond_from_last_token=cfg["cfg"]["uncond_from_last_token"],
    )
    rows = token_reranking_table(
        sampler=sampler,
        prompt=cfg["visualize"]["prompt"],
        num_steps=cfg["visualize"]["num_generation_steps"],
        top_k=cfg["visualize"]["top_k_per_step"],
        bottom_k=cfg["visualize"]["bottom_k_per_step"],
        negative_prompt=cfg["visualize"].get("negative_prompt") or None,
    )
    out = [r.__dict__ for r in rows]
    (out_dir / "visualize.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
DISPATCH = {
    "zero_shot": _run_zero_shot,
    "cot": _run_cot,
    "humaneval": _run_humaneval,
    "flops": _run_flops,
    "entropy": _run_entropy,
    "ppl_corr": _run_ppl_corr,
    "visualize": _run_visualize,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--task", required=True, choices=list(DISPATCH.keys()))
    p.add_argument(
        "--output_dir", default=None, help="Override output_dir from config."
    )
    args = p.parse_args()

    cfg = _load_yaml(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    os.makedirs(cfg.get("output_dir", "results"), exist_ok=True)

    DISPATCH[args.task](cfg)


if __name__ == "__main__":
    main()
