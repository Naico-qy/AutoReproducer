"""Base PTLM wrapper (BART0 / FLAN-T5) and LoRA wrapping.

Per §4.1 the paper experiments with three encoder-decoder LMs:

    BART0_Large           ~ 400M params  (yuchenlin/BART0)
    FLAN-T5_Large         ~ 780M params  (google/flan-t5-large)
    FLAN-T5_3B            ~ 3B   params  (google/flan-t5-xl)

For LoRA, per addendum:
    target_modules = ['q', 'v']    # query & value of self-attention
    r=16, alpha=32, dropout=0.1, bias='none', task_type=SEQ_2_SEQ_LM
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
    )
except Exception:  # pragma: no cover - kept importable on minimal envs
    torch = None
    nn = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None


# ---------------------------------------------------------------------
@dataclass
class BaseLM:
    """Wraps a HuggingFace seq2seq LM + tokenizer + name.

    Mirrors the notation `f_0` from the paper.
    """

    name: str
    model: object
    tokenizer: object
    type_: str  # 'bart' or 't5'

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def train(self):
        self.model.train()
        return self

    @torch.no_grad() if torch is not None else (lambda f: f)
    def generate(self, x: str, max_new_tokens: int = 64) -> str:
        device = next(self.model.parameters()).device
        enc = self.tokenizer(
            x,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


# ---------------------------------------------------------------------
def load_base_lm(
    name: str = "google/flan-t5-large",
    type_: str = "t5",
    dtype: str = "auto",
) -> BaseLM:
    """Load tokenizer + model from HuggingFace."""
    if AutoModelForSeq2SeqLM is None:
        raise RuntimeError("transformers not installed")

    torch_dtype = None
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        name,
        torch_dtype=torch_dtype,
    )
    return BaseLM(name=name, model=model, tokenizer=tokenizer, type_=type_)


# ---------------------------------------------------------------------
def wrap_with_lora(
    base_lm: BaseLM,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.1,
    target_modules=("q", "v"),
    bias: str = "none",
):
    """Wrap a base seq2seq LM with LoRA per the addendum.

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1,
        bias='none', target_modules=['q', 'v'],
    )
    """
    from peft import LoraConfig, TaskType, get_peft_model

    cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        target_modules=list(target_modules),
    )
    base_lm.model = get_peft_model(base_lm.model, cfg)
    return base_lm


# ---------------------------------------------------------------------
def freeze_all_but_lm_head(base_lm: BaseLM) -> BaseLM:
    """Head-only fine-tuning setup (§4.1).

    Only the output LM head (lm_head / final embedding tying) is trainable.
    """
    if torch is None:
        raise RuntimeError("PyTorch not installed")
    for p in base_lm.model.parameters():
        p.requires_grad_(False)
    head = getattr(base_lm.model, "lm_head", None)
    if head is None:
        # T5 ties weights to shared embeddings; expose those instead
        head = getattr(base_lm.model, "shared", None)
    if head is not None:
        for p in head.parameters():
            p.requires_grad_(True)
    return base_lm


def n_trainable_params(base_lm: BaseLM) -> int:
    return sum(p.numel() for p in base_lm.model.parameters() if p.requires_grad)
