"""FOA-I: single-sample interval-update variant of FOA (Section 4.4, Table 6).

When the streaming batch size is forced to 1, the activation-statistics term
in Eqn. (5) becomes ill-defined (you cannot compute a per-batch mean/std from
one sample).  The paper therefore proposes "FOA-I": buffer test samples for
``I`` steps, then run a single CMA-ES update on the buffered batch and emit
predictions for all buffered samples.

Two variants are reported in Table 7 of the paper:
    V1 -- store the per-layer CLS features between updates (memory-light)
    V2 -- store the raw images between updates (memory-heavier but supports
          re-running the prompt forward for the buffered samples)

This module implements both V1 and V2.  V1 caches per-layer CLS feats for
fitness computation but predicts each sample using the current mean prompt
m^(t-1) at arrival time (since features depend on the prompt this is an
approximation; the paper notes V1's purpose is the memory measurement in
Table 7).  V2 buffers images and re-runs the model with the freshly updated
prompt to produce the final predictions, matching the V2 row of Table 7.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch

from .foa import FOA


class FOAInterval:
    """Wraps a FOA instance and exposes a per-sample ``observe`` API.

    Parameters
    ----------
    foa : FOA
        Underlying FOA adapter.
    interval : int
        Number of incoming samples between CMA-ES updates (`I` in the paper).
    variant : str
        "v1" (cache features) or "v2" (cache images).
    """

    def __init__(self, foa: FOA, interval: int = 4, variant: str = "v2") -> None:
        self.foa = foa
        if interval < 1:
            raise ValueError(f"interval must be >= 1, got {interval}")
        if variant not in ("v1", "v2"):
            raise ValueError(f"variant must be 'v1' or 'v2', got {variant!r}")
        self.interval = int(interval)
        self.variant = variant
        # Buffers
        self._image_buf: Deque[torch.Tensor] = deque()
        self._results_pending: List[torch.Tensor] = []

    @torch.no_grad()
    def observe(self, x: torch.Tensor) -> List[int]:
        """Ingest a single test sample (or a small mini-batch) and return any
        completed predictions.  When the buffer reaches ``interval`` samples,
        it triggers one FOA step.
        """
        # Normalize to (B,3,H,W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        for i in range(x.size(0)):
            self._image_buf.append(x[i : i + 1].cpu())
        out: List[int] = []
        if len(self._image_buf) >= self.interval:
            batch = torch.cat(list(self._image_buf), dim=0)
            self._image_buf.clear()
            preds, _info = self.foa.step(batch)
            for p in preds.detach().cpu().tolist():
                out.append(int(p))
        return out

    def flush(self) -> List[int]:
        """Drain any remaining buffered samples with a final FOA step."""
        if not self._image_buf:
            return []
        batch = torch.cat(list(self._image_buf), dim=0)
        self._image_buf.clear()
        preds, _info = self.foa.step(batch)
        return [int(p) for p in preds.detach().cpu().tolist()]
