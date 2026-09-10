"""§4.1 -- Cost analysis via ANCOVA.

The paper claims (§4):

    "across 5 out of 9 tasks, there is a statistically insignificant
     difference between using CFG and using vanilla prompting with a model
     of twice the size at p = .01, according to ANCOVA regression analysis"

We implement that claim with `statsmodels`, regressing benchmark accuracy
on `log(FLOPs)` and a binary `cfg_treatment` factor.  The factor's
coefficient and p-value answer the question "for the same compute budget,
does CFG outperform a 2× larger vanilla model?"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
except ImportError:  # pragma: no cover
    sm = None  # type: ignore
    ols = None  # type: ignore

from model.flops import (
    ElectraFlopsConfig,
    count_flops_electra,
    hf_config_to_electra,
)


# ---------------------------------------------------------------------------
# Build the per-task cost table
# ---------------------------------------------------------------------------
def build_cost_table(
    accuracies: Dict[str, Dict[str, float]],  # acc[model][task]
    flops: Dict[str, int],  # flops[model]
    cfg_models: List[str],  # which rows are CFG-treated
) -> pd.DataFrame:
    """Build a long-form DataFrame for ANCOVA.

    Columns: model, task, accuracy, log_flops, cfg_treatment (0/1).
    """
    rows = []
    for model, task_scores in accuracies.items():
        for task, acc in task_scores.items():
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "accuracy": float(acc),
                    "log_flops": float(np.log(flops[model])),
                    "cfg_treatment": int(model in cfg_models),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ANCOVA per-task
# ---------------------------------------------------------------------------
@dataclass
class AncovaResult:
    task: str
    coef_cfg: float
    p_cfg: float
    significant: bool


def run_ancova(df: pd.DataFrame, alpha: float = 0.01) -> List[AncovaResult]:
    """Run an ANCOVA per task, controlling for log-FLOPs.

    Returns one result per task.  `significant=True` means the CFG vs.
    vanilla difference is statistically significant at the given alpha.
    """
    if ols is None:  # pragma: no cover
        raise RuntimeError("statsmodels is required for ANCOVA.")
    out: List[AncovaResult] = []
    for task, sub in df.groupby("task"):
        if sub["cfg_treatment"].nunique() < 2:
            continue
        model = ols("accuracy ~ log_flops + C(cfg_treatment)", data=sub).fit()
        coef = float(model.params.get("C(cfg_treatment)[T.1]", 0.0))
        pval = float(model.pvalues.get("C(cfg_treatment)[T.1]", 1.0))
        out.append(
            AncovaResult(
                task=task,
                coef_cfg=coef,
                p_cfg=pval,
                significant=pval < alpha,
            )
        )
    return out


def estimate_inference_cost(hf_config, with_cfg: bool = False) -> int:
    """Per-token FLOPs for a given HF config (cf. §4 / Table 6 helper)."""
    cfg = hf_config_to_electra(hf_config)
    return count_flops_electra(cfg, with_cfg=with_cfg)
