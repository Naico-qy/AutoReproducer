"""§5.2 -- Comparing CFG vs. Instruction-Tuning by perplexity.

The paper (and Figure 5a) reports Pearson correlations between three
per-prompt perplexities computed on the **continuation** only (addendum):

    PPL p(y|x)   -- vanilla Falcon-7b-Base
    PPL cfg      -- Falcon-7b-Base + CFG with γ=1.5
    PPL instruct -- Falcon-7b-Instruct

Reported correlations (Figure 5a):
    ρ(p(y|x), cfg)      = 0.94
    ρ(p(y|x), instruct) = 0.83
    ρ(cfg, instruct)    = 0.70
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data.p3_sampler import P3SamplerConfig, sample_p3
from model.architecture import CFGSampler
from model.entropy import continuation_perplexity, cfg_continuation_perplexity


@dataclass
class PPLCorrelation:
    n: int
    rho_baseline_cfg: float
    rho_baseline_instruct: float
    rho_cfg_instruct: float
    df: pd.DataFrame  # raw per-prompt PPLs


@torch.no_grad()
def perplexity_correlation(
    base_model,
    instruct_model,
    tokenizer,
    p3_cfg: P3SamplerConfig,
    cfg_gamma: float = 1.5,
    max_samples: int = 5_000,
) -> PPLCorrelation:
    """Compute the three perplexities and their pairwise Pearson correlations.

    The continuation we condition on is `target` (the P3 reference completion).
    """
    base_sampler = CFGSampler(
        model=base_model,
        tokenizer=tokenizer,
        guidance_scale=cfg_gamma,
        uncond_from_last_token=True,
    )

    rows = []
    n = 0
    for sample in tqdm(
        sample_p3(p3_cfg, tokenizer=tokenizer), total=max_samples, desc="ppl"
    ):
        if n >= max_samples:
            break
        prompt = sample["input"]
        target = sample["target"]
        if not target:
            continue
        ppl_base = continuation_perplexity(base_model, tokenizer, prompt, target)
        ppl_inst = continuation_perplexity(instruct_model, tokenizer, prompt, target)
        ppl_cfg = cfg_continuation_perplexity(base_sampler, prompt, target)
        rows.append(
            {"ppl_base": ppl_base, "ppl_cfg": ppl_cfg, "ppl_instruct": ppl_inst}
        )
        n += 1

    df = pd.DataFrame(rows).dropna()
    if len(df) < 2:
        return PPLCorrelation(
            n=len(df),
            rho_baseline_cfg=float("nan"),
            rho_baseline_instruct=float("nan"),
            rho_cfg_instruct=float("nan"),
            df=df,
        )
    rho_bc = float(df["ppl_base"].corr(df["ppl_cfg"]))
    rho_bi = float(df["ppl_base"].corr(df["ppl_instruct"]))
    rho_ci = float(df["ppl_cfg"].corr(df["ppl_instruct"]))
    return PPLCorrelation(
        n=len(df),
        rho_baseline_cfg=rho_bc,
        rho_baseline_instruct=rho_bi,
        rho_cfg_instruct=rho_ci,
        df=df,
    )
