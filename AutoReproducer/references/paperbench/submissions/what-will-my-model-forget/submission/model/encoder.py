"""Encoder h(x, y) used by the §3.2 / §3.3 forecasters.

Per §3.2:
    h: (x, y) ↦ R^{T × d}
    "We implement h with a trainable LM and extract its representation
     of output tokens in the final layer as h(x, y)."

Per §3.3:
    h is the *averaged* representation of all tokens in (x, y).

We therefore expose two pooling modes:
    - 'token'  : per-output-token representations of shape (T, d)
                 (used by the logit-based forecaster, §3.2)
    - 'mean'   : single (d,) vector averaged over tokens
                 (used by the representation-based forecaster, §3.3)
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover
    torch = None
    nn = None
    AutoModel = None
    AutoTokenizer = None


@dataclass
class EncoderConfig:
    name: str = "google/flan-t5-base"
    proj_dim: int = 256
    max_input_len: int = 512
    max_output_len: int = 64


class Encoder(nn.Module if nn is not None else object):
    """Trainable LM encoder + linear projection used as h(x, y).

    For a T5/BART backbone we feed the concatenation of input and output
    text into the encoder and read the final layer hidden states.
    """

    def __init__(self, cfg: EncoderConfig):
        if nn is None:
            raise RuntimeError("PyTorch / transformers not installed")
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.name)
        self.backbone = AutoModel.from_pretrained(cfg.name)

        hidden = self._infer_hidden_size()
        self.proj = nn.Linear(hidden, cfg.proj_dim)

    def _infer_hidden_size(self) -> int:
        cfg = self.backbone.config
        return getattr(cfg, "d_model", None) or getattr(cfg, "hidden_size", 768)

    # -----------------------------------------------------------------
    def forward(
        self,
        x: str,
        y: str,
        pool: str = "token",
    ):
        """Compute h(x, y).

        Args:
            x, y: input / output strings.
            pool: 'token' returns (T, d) per-token reps for §3.2;
                  'mean'  returns (d,)  averaged rep for §3.3.
        """
        if torch is None:
            raise RuntimeError("PyTorch not installed")

        device = next(self.parameters()).device
        # We feed `x [SEP] y` to the encoder side and take its hidden states.
        # This works for both BART and T5 since both expose `.encoder`.
        text = f"{x}\n{y}"
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_input_len + self.cfg.max_output_len,
        ).to(device)

        encoder = getattr(self.backbone, "encoder", self.backbone)
        out = encoder(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            return_dict=True,
        )
        hidden = out.last_hidden_state[0]  # (L, H)
        proj = self.proj(hidden)  # (L, d)

        if pool == "mean":
            mask = enc.attention_mask[0].float().unsqueeze(-1)  # (L, 1)
            return (proj * mask).sum(0) / mask.sum().clamp(min=1.0)

        # 'token' mode: only return the tail tokens corresponding to y.
        # We approximate by returning the last `max_output_len` projected
        # tokens, padded / truncated to `cfg.max_output_len`.
        T = self.cfg.max_output_len
        rep = proj[-T:] if proj.size(0) >= T else _pad_to(proj, T)
        return rep  # (T, d)


def _pad_to(t, T):
    pad = T - t.size(0)
    if pad <= 0:
        return t[-T:]
    return torch.cat(
        [torch.zeros(pad, t.size(1), device=t.device, dtype=t.dtype), t], dim=0
    )


def build_encoder(cfg: EncoderConfig | None = None) -> Encoder:
    return Encoder(cfg or EncoderConfig())
