"""Build the 24,576 pairwise (prompt, y+, y-) preference dataset.

Section 4.2 of the paper:

    "We build our pairwise toxicity dataset using PPLM (Dathathri et al., 2019).
     For each prompt, we generate a positive sample using greedy sampling with
     GPT2, while using PPLM to generate negative (toxic) samples.  We use our
     toxic probe W_toxic as our attribute classifier to guide towards toxic
     outputs.  We create 24,576 pairs of toxic and nontoxic continuations."

Outputs JSONL with one record per line:
    {"prompt": "...", "chosen": "y+ ...", "rejected": "y- ..."}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from data import load_wikitext_prompts
from model.architecture import GPT2WithHooks, LinearToxicityProbe
from model.pplm import PPLMConfig, generate_greedy, generate_pplm_toxic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, build only this many pairs (for smoke runs).",
    )
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_pairs = args.limit if args.limit is not None else cfg["pplm"]["num_pairs"]
    gen_len = cfg["pplm"]["generation_length"]

    # 1. load model and probe
    print("Loading model + probe...")
    model = GPT2WithHooks(cfg["model"]["name"]).to(device)
    model.eval()
    probe_path = Path(cfg["output"]["probe_ckpt"])
    probe = LinearToxicityProbe(model.d_model).to(device)
    if probe_path.exists():
        ckpt = torch.load(probe_path, map_location=device)
        probe.load_state_dict(ckpt["state_dict"])
    probe.eval()

    # 2. load Wikitext prompts (Section 4.2 says Wikitext-2 is the prompt source)
    prompts = load_wikitext_prompts(n_prompts=n_pairs, split="train")
    print(f"Got {len(prompts)} Wikitext prompts; building pairs...")

    pplm_cfg = PPLMConfig(
        step_size=cfg["pplm"]["pplm_step_size"],
        num_iterations=cfg["pplm"]["pplm_num_iterations"],
        window_length=cfg["pplm"]["pplm_window_length"],
        horizon_length=cfg["pplm"]["pplm_horizon_length"],
        kl_scale=cfg["pplm"]["pplm_kl_scale"],
        gm_scale=cfg["pplm"]["pplm_gm_scale"],
        gamma=cfg["pplm"]["pplm_gamma"],
        target_class=1,
    )

    out_path = Path(cfg["output"]["pairwise_data"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. for each prompt, greedy y+ and PPLM y-
    with open(out_path, "w") as fout:
        for prompt in tqdm(prompts[:n_pairs], desc="build pairs"):
            try:
                y_pos = generate_greedy(model, model.tokenizer, prompt, gen_len, device)
                y_neg = generate_pplm_toxic(
                    model, model.tokenizer, probe, prompt, gen_len, device, pplm_cfg
                )
            except Exception as e:  # noqa: BLE001
                # Skip prompts that fail (e.g. very long after BPE)
                print(f"skip: {e}")
                continue
            rec = {"prompt": prompt, "chosen": y_pos, "rejected": y_neg}
            fout.write(json.dumps(rec) + "\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
