"""
Output Mapping f_out for Visual Reprogramming (Section 2.3, Appendix A.4).

Implements three label-mapping strategies, all non-parametric (Chen et al., 2023):
- Random Label Mapping (Rlm)        -- Eq. (2)
- Frequent Label Mapping (Flm)      -- Algorithm 3 (Eq. (12))
- Iterative Label Mapping (Ilm)     -- Algorithm 4

The mapping is a one-to-one injective function from a subset Y_sub^P of size
|Y^T| of pre-trained labels to target labels. Implemented as an integer
permutation `pred2target[k_T]` -> source-class index in Y^P, equivalently a
column-selecting matmul on logits.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class LabelMapping(nn.Module):
    """Holds a mapping from |Y^P|-dim logits to |Y^T|-dim logits.

    The mapping is stored as an index buffer ``y_sub`` of shape [k_T] -- the
    chosen pre-trained class indices, in order of target labels 0..k_T-1.
    f_out(logits)[:, t] = logits[:, y_sub[t]].
    """

    def __init__(self, num_target: int, num_source: int = 1000) -> None:
        super().__init__()
        self.num_target = num_target
        self.num_source = num_source
        # Initialise with first k_T pre-trained classes -- overwritten by Rlm/Flm/Ilm.
        idx = torch.arange(num_target, dtype=torch.long)
        self.register_buffer("y_sub", idx)

    def forward(self, logits_p: torch.Tensor) -> torch.Tensor:
        # Select the columns of logits corresponding to y_sub
        return logits_p.index_select(dim=1, index=self.y_sub)

    # ------------------------------------------------------------------
    # Mapping update procedures
    # ------------------------------------------------------------------
    def random_mapping(self, seed: int = 0) -> None:
        """Rlm (Eq. 2): randomly permute pre-trained class indices and pick the
        first k_T as injective mapping."""
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.num_source)[: self.num_target]
        self.y_sub = torch.tensor(perm, dtype=torch.long, device=self.y_sub.device)

    @staticmethod
    def _frequency_distribution(
        model: nn.Module, loader, device: torch.device, num_source: int, num_target: int
    ) -> torch.Tensor:
        """Algorithm 2: count d[y^P, y^T] = #{predicted as y^P given true y^T}."""
        d = torch.zeros(num_source, num_target, dtype=torch.long, device=device)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                preds = logits.argmax(dim=1)
                # Vectorised scatter-add into 2D histogram
                idx = preds * num_target + y
                ones = torch.ones_like(idx, dtype=torch.long)
                d.view(-1).scatter_add_(0, idx, ones)
        if was_training:
            model.train()
        return d

    def frequent_mapping(self, model: nn.Module, loader, device: torch.device) -> None:
        """Flm (Algorithm 3): greedy maximum-frequency injective mapping."""
        d = self._frequency_distribution(
            model, loader, device, self.num_source, self.num_target
        )
        d = d.clone()
        y_sub = torch.full((self.num_target,), -1, dtype=torch.long, device=device)
        for _ in range(self.num_target):
            flat = d.argmax()
            yp = (flat // self.num_target).item()
            yt = (flat % self.num_target).item()
            y_sub[yt] = yp
            d[yp, :] = 0  # avoid re-using yp
            d[:, yt] = 0  # avoid re-assigning yt
        self.y_sub = y_sub

    def iterative_mapping(self, model: nn.Module, loader, device: torch.device) -> None:
        """Ilm (Algorithm 4): per-iteration update -- same as Flm but called
        once per epoch with the *current* f_in. The implementation differs
        from Flm only in how often it is invoked (handled by train.py).
        """
        self.frequent_mapping(model, loader, device)


def build_label_mapping(
    num_target: int, mapping_method: str = "ilm", seed: int = 0
) -> LabelMapping:
    """Factory."""
    lm = LabelMapping(num_target=num_target, num_source=1000)
    if mapping_method == "rlm":
        lm.random_mapping(seed=seed)
    # 'flm' / 'ilm' are populated externally before/during training.
    return lm
