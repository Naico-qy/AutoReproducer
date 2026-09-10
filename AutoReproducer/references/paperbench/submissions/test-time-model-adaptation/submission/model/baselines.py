"""Baseline test-time adaptation methods used as reference points in the
paper: NoAdapt, TENT, T3A.

Authoritative references (verified via paper_search/CrossRef during this
session):

- TENT: Wang, D.; Shelhamer, E.; Liu, S.; Olshausen, B.; Darrell, T.
  "Tent: Fully Test-Time Adaptation by Entropy Minimization." ICLR 2021.
  https://openreview.net/forum?id=uXl3bZLkr3c

- T3A: Iwasawa, Y.; Matsuo, Y. "Test-Time Classifier Adjustment Module for
  Model-Agnostic Domain Generalization." NeurIPS 2021.

These are minimal but faithful reimplementations -- the FOA paper says
"Baselines should be imported from the links provided in the paper"
(addendum), but our submission must be self-contained, so we provide
canonical implementations.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .architecture import PromptedViT


# ----------------------------------------------------------------------
# Common evaluator
# ----------------------------------------------------------------------
class _BaseAdapter:
    name: str = "base"

    def step(self, x: torch.Tensor):  # pragma: no cover
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, loader, device: torch.device) -> Dict[str, float]:
        n_correct = 0
        n_total = 0
        bins = np.zeros((15, 3), dtype=np.float64)
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch["image"], batch["label"]
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.set_grad_enabled(getattr(self, "needs_grad", False)):
                logits = self.step(x)
            preds = logits.argmax(dim=-1)
            n_correct += int((preds == y).sum().item())
            n_total += int(y.numel())
            probs = F.softmax(logits, dim=-1)
            conf, _ = probs.max(dim=-1)
            for c, p, t in zip(
                conf.detach().cpu().numpy(),
                preds.detach().cpu().numpy(),
                y.detach().cpu().numpy(),
            ):
                b = min(int(float(c) * 15), 14)
                bins[b, 0] += 1.0
                bins[b, 1] += float(c)
                bins[b, 2] += float(p == t)
        acc = 100.0 * n_correct / max(n_total, 1)
        ece = 0.0
        if n_total > 0:
            for b in range(15):
                if bins[b, 0] == 0:
                    continue
                avg_conf = bins[b, 1] / bins[b, 0]
                avg_acc = bins[b, 2] / bins[b, 0]
                ece += (bins[b, 0] / n_total) * abs(avg_conf - avg_acc)
            ece *= 100.0
        return {"accuracy": acc, "ece": ece, "n_total": float(n_total)}


# ----------------------------------------------------------------------
# NoAdapt: just the source model.
# ----------------------------------------------------------------------
class NoAdapt(_BaseAdapter):
    name = "noadapt"
    needs_grad = False

    def __init__(self, model: PromptedViT) -> None:
        self.model = model
        self.model.eval()
        zero_p = torch.zeros(model.num_prompts, model.embed_dim)
        self._zero_prompt = zero_p

    @torch.no_grad()
    def step(self, x: torch.Tensor) -> torch.Tensor:
        device = next(self.model.parameters()).device
        return self.model(x, prompt_override=self._zero_prompt.to(device))


# ----------------------------------------------------------------------
# TENT: optimize affine params of LayerNorm via entropy minimization.
# ----------------------------------------------------------------------
class TENT(_BaseAdapter):
    name = "tent"
    needs_grad = True

    def __init__(self, model: PromptedViT, lr: float = 1e-3) -> None:
        self.model = model
        # Collect affine parameters of LayerNorms. ViT uses LayerNorm.
        self._params: List[nn.Parameter] = []
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    m.weight.requires_grad_(True)
                    self._params.append(m.weight)
                if m.bias is not None:
                    m.bias.requires_grad_(True)
                    self._params.append(m.bias)
        for p in model.parameters():
            if p is not model.prompt and p not in self._params:
                p.requires_grad_(False)
        # SGD per the paper's TENT description (Table 9 row "TENT").
        self.opt = torch.optim.SGD(self._params, lr=lr, momentum=0.9)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        self.model.train()  # enable LayerNorm updates conceptually
        # Use zero prompt -- TENT does not adapt prompts.
        device = next(self.model.parameters()).device
        zero_p = torch.zeros(
            self.model.num_prompts, self.model.embed_dim, device=device
        )
        logits = self.model(x, prompt_override=zero_p)
        # entropy minimization
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(probs * log_probs).sum(dim=-1).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return logits.detach()


# ----------------------------------------------------------------------
# T3A: prototype-based classifier adjustment (Iwasawa & Matsuo 2021).
# ----------------------------------------------------------------------
class T3A(_BaseAdapter):
    """Builds class prototypes online from the model's pre-classifier feature
    e_N^0 weighted by softmax confidence; prediction = argmax cosine similarity.

    M is the per-class support set size (default 100).
    """

    name = "t3a"
    needs_grad = False

    def __init__(
        self, model: PromptedViT, num_classes: int = 1000, M: int = 100
    ) -> None:
        self.model = model
        self.model.eval()
        self.num_classes = int(num_classes)
        self.M = int(M)
        # We seed prototypes with the head's weight rows -- the "trained
        # classifier" prototypes (Iwasawa & Matsuo Sec. 3).
        with torch.no_grad():
            W = model.head.weight.detach().clone()  # (C, D)
        self._support = [W[c : c + 1].clone() for c in range(num_classes)]
        self._support_ent = [torch.zeros(1) for _ in range(num_classes)]

    @torch.no_grad()
    def step(self, x: torch.Tensor) -> torch.Tensor:
        device = next(self.model.parameters()).device
        zero_p = torch.zeros(
            self.model.num_prompts, self.model.embed_dim, device=device
        )
        logits, cls_final, _ = self.model(
            x, prompt_override=zero_p, return_all_cls=True
        )
        probs = F.softmax(logits, dim=-1)
        ent = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        preds = logits.argmax(dim=-1)
        # Append features to per-class supports
        for i in range(x.size(0)):
            c = int(preds[i].item())
            f = cls_final[i : i + 1].detach().cpu()
            self._support[c] = torch.cat([self._support[c], f], dim=0)
            self._support_ent[c] = torch.cat(
                [self._support_ent[c], ent[i : i + 1].detach().cpu()], dim=0
            )
            # Truncate to top-M lowest entropy
            if self._support[c].size(0) > self.M:
                _, keep = torch.topk(self._support_ent[c], self.M, largest=False)
                self._support[c] = self._support[c][keep]
                self._support_ent[c] = self._support_ent[c][keep]
        # Build prototypes -> mean of each class support
        protos = torch.stack(
            [self._support[c].mean(dim=0) for c in range(self.num_classes)],
            dim=0,
        ).to(device)
        # Cosine similarity classification
        f = F.normalize(cls_final, dim=-1)
        protos_n = F.normalize(protos, dim=-1)
        new_logits = f @ protos_n.t()
        return new_logits
