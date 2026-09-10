"""NetHack APPO actor-critic architecture, as described in App. B.1.

Architecture summary (Tuyls et al., 2023; Hambro et al., 2022c):

    map glyphs (24x80 chars + colors) ──► embedding lookup ──► ResNet
                                                                  │
    blstats (27 floats)               ──► 2-layer MLP ────────────┤
                                                                  │
    message (256 chars, ascii)        ──► 2-layer MLP ────────────┤
                                                                  ▼
                                                           concat & flatten
                                                                  │
                                                                LSTM
                                                                  │
                                                          ┌───────┴───────┐
                                                          ▼               ▼
                                                     policy head     baseline head

The hyperparameters in `configs/default.yaml` (Table 1) reproduce the 30 M-parameter
"Scaled-BC" model from Tuyls et al. (2023):

* hidden_dim = 1738       (LSTM hidden + projection size)
* activation = relu
* baseline_cost = 1
* unroll_length = 32
* batch_size = 128
* discounting = 0.999999
* APPO clip_policy = 0.1, clip_baseline = 1.0
* AdamW lr = 1e-4, weight_decay = 1e-4, eps = 1e-7
* grad_norm_clipping = 4
* entropy_cost = 0.001 (turned off when knowledge retention is on)

Reference verified via paper_search:
  Tuyls, J., Madeka, D., Torkkola, K., Foster, D., Narasimhan, K., Kakade, S.
  "Scaling Laws for Imitation Learning in NetHack."
  arXiv:2307.09423, 2023. (S2 / arXiv ID confirmed.)
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- glyph + character embeddings -----------------------------------------

NUM_CHARS = 256  # ascii chars on the dungeon screen
NUM_COLORS = 16
EMBED_DIM = 32


class ResBlock(nn.Module):
    """3x3 conv residual block."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(F.relu(x)))
        h = self.conv2(h)
        return x + h


class MainScreenEncoder(nn.Module):
    """Embed dungeon screen (chars+colors) and pass through 3 residual blocks."""

    def __init__(self, channels: int = 16, out_dim: int = 512):
        super().__init__()
        self.char_emb = nn.Embedding(NUM_CHARS, EMBED_DIM)
        self.color_emb = nn.Embedding(NUM_COLORS, EMBED_DIM)
        self.stem = nn.Conv2d(2 * EMBED_DIM, channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(3)])
        self.out_dim = out_dim
        self.proj = nn.Linear(channels * 21 * 79, out_dim)

    def forward(self, chars: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        # chars/colors: (B, 21, 79) long
        c = self.char_emb(chars)  # (B, H, W, E)
        col = self.color_emb(colors)  # (B, H, W, E)
        x = torch.cat([c, col], dim=-1)  # (B, H, W, 2E)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.relu(self.stem(x))
        for blk in self.blocks:
            x = blk(x)
        x = x.flatten(1)
        return F.relu(self.proj(x))


class MLPEncoder(nn.Module):
    """Two-layer MLP for blstats and message embeddings (App. B.1)."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc2(F.relu(self.fc1(x))))


class NetHackNet(nn.Module):
    """LSTM-based actor-critic over the three NLE observation streams."""

    NUM_ACTIONS = 121  # NLE action set (with extras)

    def __init__(
        self,
        hidden_dim: int = 1738,
        num_actions: int = NUM_ACTIONS,
        blstats_dim: int = 27,
        message_dim: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        self.screen_encoder = MainScreenEncoder(out_dim=512)
        self.blstats_encoder = MLPEncoder(blstats_dim, hidden_dim=128, out_dim=128)
        self.message_encoder = MLPEncoder(message_dim, hidden_dim=128, out_dim=128)

        merged = (
            self.screen_encoder.out_dim
            + self.blstats_encoder.out_dim
            + self.message_encoder.out_dim
        )
        self.fc_merge = nn.Linear(merged, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1)

        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.baseline_head = nn.Linear(hidden_dim, 1)

    # ---- backbone ----------------------------------------------------------

    def encode(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        s = self.screen_encoder(obs["chars"], obs["colors"])
        b = self.blstats_encoder(obs["blstats"].float())
        m = self.message_encoder(obs["message"].float())
        x = torch.cat([s, b, m], dim=-1)
        return F.relu(self.fc_merge(x))

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        prev_state=None,
        done_mask: torch.Tensor | None = None,
    ):
        """Run one step (or one unroll) through the LSTM.

        `obs` tensors are shaped (T, B, ...) for unrolls or (B, ...) for a
        single step. `done_mask` is (T, B) and resets the LSTM hidden state
        on episode boundaries (1.0 → keep, 0.0 → reset).
        """
        if obs["chars"].dim() == 3:
            # single time-step – add T dim
            obs = {k: v.unsqueeze(0) for k, v in obs.items()}
            single_step = True
        else:
            single_step = False

        T, B = obs["chars"].shape[:2]
        flat_obs = {k: v.reshape((T * B,) + v.shape[2:]) for k, v in obs.items()}
        feats = self.encode(flat_obs).view(T, B, self.hidden_dim)

        if prev_state is None:
            h = torch.zeros(1, B, self.hidden_dim, device=feats.device)
            c = torch.zeros(1, B, self.hidden_dim, device=feats.device)
            prev_state = (h, c)

        if done_mask is None:
            outs, new_state = self.lstm(feats, prev_state)
        else:
            outs = []
            h, c = prev_state
            for t in range(T):
                m = done_mask[t].view(1, B, 1)
                h = h * m
                c = c * m
                out, (h, c) = self.lstm(feats[t : t + 1], (h, c))
                outs.append(out)
            outs = torch.cat(outs, dim=0)
            new_state = (h, c)

        logits = self.policy_head(outs)  # (T, B, A)
        baseline = self.baseline_head(outs).squeeze(-1)  # (T, B)

        if single_step:
            logits = logits.squeeze(0)
            baseline = baseline.squeeze(0)
        return logits, baseline, new_state

    # ---- conveniences ------------------------------------------------------

    def actor_parameters(self):
        """Return params used for retention losses (everything except baseline)."""
        for name, p in self.named_parameters():
            if "baseline_head" not in name:
                yield p

    def freeze_encoders(self):
        for mod in (self.screen_encoder, self.blstats_encoder, self.message_encoder):
            for p in mod.parameters():
                p.requires_grad = False
