"""
BBox-Adapter: scalar energy model g_theta(x, y).

Implements the parameterisation of Eq. (1) of:

    Sun, Zhuang, Wei, Zhang, Dai.
    "BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models."
    ICML 2024.  arXiv:2402.08219.

In Eq. (1) the adapted distribution is

    p_theta(y | x) = p_LLM(y | x) * exp( g_theta(x, y) ) / Z_theta(x)

where g_theta is a scalar-valued energy function realised here by a
DeBERTa-v3 (or BERT) encoder followed by a linear projection of the
[CLS] hidden state (Appendix H.2).

The black-box LLM p_LLM is **frozen** and never appears in this module —
its outputs enter the energy computation only as concatenated text in `y`.

Reference verified via CrossRef / GPT (no DOI, arXiv preprint 2110.14168
for GSM8K, used as one of the four target tasks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


@dataclass
class AdapterConfig:
    backbone: str = "microsoft/deberta-v3-base"
    hidden_dim: int = 768
    max_length: int = 512
    dropout: float = 0.1
    pooling: str = "cls"  # cls | mean


class BBoxAdapter(nn.Module):
    """
    The scoring (energy) head g_theta(x, y) used by BBox-Adapter.

    The encoder consumes `[x ; y]` (question concatenated with the
    candidate answer) and returns a scalar energy.  Following Appendix
    H.2, the default backbone is `microsoft/deberta-v3-base` (~86M
    parameters, 0.1B variant) or `microsoft/deberta-v3-large` (~304M,
    0.3B variant).

    The energy head is a 2-layer MLP with tanh activation; the final
    layer outputs a single scalar (no activation).  Tanh is used to
    keep the latent close to the bounded range used by EBM scoring
    heads in (Du & Mordatch, 2019), which the paper cites for spectral
    regularisation.
    """

    def __init__(self, cfg: AdapterConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(cfg.backbone)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"

        h = cfg.hidden_dim
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, 1),
        )

        # Reasonable Xavier init for the scalar head (energies should
        # start near zero — important for the NCE log-sum-exp).
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------
    def encode_pair(
        self,
        questions: List[str],
        answers: List[str],
        device: Optional[torch.device] = None,
    ):
        """Tokenise (x, y) pairs as a single packed sequence."""
        assert len(questions) == len(answers)
        enc = self.tokenizer(
            questions,
            answers,
            padding=True,
            truncation="only_second",
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )
        if device is not None:
            enc = {k: v.to(device) for k, v in enc.items()}
        return enc

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return energy g_theta(x, y) of shape (batch,)."""
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        out = self.encoder(**kwargs)
        h = out.last_hidden_state  # (B, T, H)

        if self.cfg.pooling == "cls":
            pooled = h[:, 0]  # CLS token
        elif self.cfg.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        else:
            raise ValueError(f"unknown pooling: {self.cfg.pooling}")

        return self.head(pooled).squeeze(-1)

    # ------------------------------------------------------------------
    # Convenience: score a list of (x, y) text pairs in one call.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def score(
        self,
        questions: List[str],
        answers: List[str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Eval-mode scoring used by sentence_beam_search."""
        was_training = self.training
        self.eval()
        enc = self.encode_pair(questions, answers, device=device)
        energies = self.forward(**enc)
        if was_training:
            self.train()
        return energies

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save({"state_dict": self.state_dict(), "cfg": self.cfg.__dict__}, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "BBoxAdapter":
        ckpt = torch.load(path, map_location=map_location)
        cfg = AdapterConfig(**ckpt["cfg"])
        model = cls(cfg)
        model.load_state_dict(ckpt["state_dict"])
        return model

    # ------------------------------------------------------------------
    # Parameter count helper (paper reports 0.1B / 0.3B)
    # ------------------------------------------------------------------
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
