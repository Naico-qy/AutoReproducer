"""Training loops for the three forecasters.

Algorithms summarised:

    train_threshold:  count z_ij in D_R^Train + tune γ to max F1.
    train_logit:      Algorithm 1 (Appendix F) — margin loss on Eqn. 2.
    train_repr:       Algorithm 3 (Appendix F) — BCE on σ(h_j · h_i + b_j).
"""

from __future__ import annotations

from typing import Iterable

try:
    import torch
    from torch.optim import AdamW
except Exception:  # pragma: no cover
    torch = None
    AdamW = None

from model.threshold import ThresholdForecaster
from model.logit_forecaster import LogitForecaster, LogitForecasterConfig
from model.repr_forecaster import (
    RepresentationForecaster,
    ReprForecasterConfig,
    frequency_prior,
)
from model.encoder import EncoderConfig
from model.losses import binary_ce_loss


# ---------------------------------------------------------------------
def train_threshold(
    train_pairs: list[tuple[str, str, int]],
    min_g: int = 1,
    max_g: int = 200,
) -> ThresholdForecaster:
    """§3.1 Algorithm: tune γ to maximize F1 on D_R^Train."""
    f = ThresholdForecaster()
    f.tune_gamma(train_pairs, min_g=min_g, max_g=max_g)
    return f


# ---------------------------------------------------------------------
def train_logit(
    train_examples: list[dict],
    base_lm,
    cache_f0_xj,  # uid_j -> tensor (T, V) of f̂_0(x_j)
    cache_delta_xi,  # uid_i -> tensor (T, V) of f̂_i(x_i) - f̂_0(x_i)
    epochs: int = 5,
    lr: float = 1e-4,
    encoder_name: str = "google/flan-t5-base",
    proj_dim: int = 256,
) -> LogitForecaster:
    """Algorithm 1: train h s.t. predicted logits separate forgotten/not."""
    if torch is None:
        raise RuntimeError("PyTorch not installed")

    enc_cfg = EncoderConfig(name=encoder_name, proj_dim=proj_dim)
    cfg = LogitForecasterConfig(proj_dim=proj_dim)
    fc = LogitForecaster(cfg=cfg, encoder_cfg=enc_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fc = fc.to(device)
    opt = AdamW(fc.parameters(), lr=lr)

    for epoch in range(epochs):
        for ex in train_examples:
            ex_i, ex_j, z_ij = ex["i"], ex["j"], int(ex["z"])
            f0_xj = cache_f0_xj[ex_j["uid"]].to(device)
            delta_xi = cache_delta_xi[ex_i["uid"]].to(device)
            forecast = fc.forecast_logits(
                ex_j["x"],
                ex_j["y"],
                ex_i["x"],
                ex_i["y"],
                delta_logits_xi=delta_xi,
                f0_logits_xj=f0_xj,
            )
            # gold token ids of y_j (truncated to T)
            T = forecast.size(0)
            yids = (
                base_lm.tokenizer(
                    ex_j["y"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=T,
                )
                .input_ids[0]
                .to(device)
            )
            yids = (
                yids[:T]
                if yids.numel() >= T
                else torch.cat(
                    [
                        yids,
                        torch.full(
                            (T - yids.numel(),),
                            base_lm.tokenizer.pad_token_id,
                            device=device,
                            dtype=yids.dtype,
                        ),
                    ]
                )
            )
            loss = fc.loss(forecast, yids, z_ij=z_ij)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return fc


# ---------------------------------------------------------------------
def train_repr(
    train_pairs_full: list[tuple[dict, dict, int]],
    encoder_name: str = "google/flan-t5-base",
    use_frequency_prior: bool = True,
    epochs: int = 5,
    lr: float = 1e-4,
) -> RepresentationForecaster:
    """Algorithm 3: train repr-based forecaster with BCE + freq-prior."""
    if torch is None:
        raise RuntimeError("PyTorch not installed")

    enc_cfg = EncoderConfig(name=encoder_name)
    cfg = ReprForecasterConfig(use_frequency_prior=use_frequency_prior)
    fc = RepresentationForecaster(cfg=cfg, encoder_cfg=enc_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fc = fc.to(device)

    # frequency prior
    flat = [(ex_i["uid"], ex_j["uid"], z) for ex_i, ex_j, z in train_pairs_full]
    fc.set_frequency_prior(frequency_prior(flat))

    opt = AdamW(fc.parameters(), lr=lr)
    for epoch in range(epochs):
        for ex_i, ex_j, z in train_pairs_full:
            score = fc(ex_i["x"], ex_i["y"], ex_j["x"], ex_j["y"], uid_j=ex_j["uid"])
            loss = binary_ce_loss(score.unsqueeze(0), [z])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return fc
