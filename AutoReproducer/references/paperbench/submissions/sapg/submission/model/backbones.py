"""
Shared backbones used by SAPG actor and critic.

The paper (Sec. 4.4 + addendum) specifies:
  * Each policy uses a SHARED backbone B_theta (actor) and C_psi (critic),
    conditioned on per-policy parameters phi_j.
  * AllegroKuka tasks: MLP encoder (768->512->256, ELU) + LSTM (1 layer, 768).
  * ShadowHand: MLP only (512->512->256->128, ELU).
  * AllegroHand: MLP only (512->256->128, ELU).

Activations are ELU (Clevert et al., 2016) per App. B.1.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn


def _act(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation {name}")


class MLPBackbone(nn.Module):
    """Plain feed-forward backbone -- used for ShadowHand / AllegroHand.

    Output dim = hidden[-1]. Phi is concatenated to the FIRST layer input
    so the shared backbone can specialise its early layers per policy.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden: List[int],
        phi_dim: int,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = obs_dim + phi_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(_act(activation))
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(
        self,
        obs: torch.Tensor,
        phi: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # phi: (..., phi_dim) broadcast against obs
        if phi.dim() < obs.dim():
            phi = phi.expand(*obs.shape[:-1], phi.shape[-1])
        x = torch.cat([obs, phi], dim=-1)
        return self.net(x), None


class LSTMBackbone(nn.Module):
    """MLP encoder followed by an LSTM -- used for AllegroKuka tasks.

    Architecture from App. B.1:
      observation -> MLP[768, 512, 256] (ELU) -> LSTM(1 layer, hidden=768)
    """

    def __init__(
        self,
        obs_dim: int,
        mlp_hidden: List[int],
        lstm_hidden: int,
        lstm_layers: int,
        phi_dim: int,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = obs_dim + phi_dim
        for h in mlp_hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(_act(activation))
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.lstm = nn.LSTM(prev, lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.out_dim = lstm_hidden

    def init_hidden(
        self, batch: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.lstm_layers, batch, self.lstm_hidden, device=device)
        c0 = torch.zeros(self.lstm_layers, batch, self.lstm_hidden, device=device)
        return h0, c0

    def forward(
        self,
        obs: torch.Tensor,
        phi: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # obs: (B, T, D) for sequence input or (B, D) for single-step
        squeezed = False
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
            squeezed = True
        if phi.dim() == 2:
            phi = phi.unsqueeze(1).expand(-1, obs.shape[1], -1)
        elif phi.dim() == 1:
            phi = phi.view(1, 1, -1).expand(obs.shape[0], obs.shape[1], -1)
        x = torch.cat([obs, phi], dim=-1)
        # Encode each timestep
        b, t, _ = x.shape
        x = self.encoder(x.reshape(b * t, -1)).reshape(b, t, -1)
        if hidden_state is None:
            hidden_state = self.init_hidden(b, x.device)
        out, hidden = self.lstm(x, hidden_state)
        if squeezed:
            out = out.squeeze(1)
        return out, hidden
