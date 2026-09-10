"""MLP-based Physics-Informed Neural Network architecture.

Reference paper:
    Rathore, P., Lei, W., Frangella, Z., Lu, L., and Udell, M. (2024).
    "Challenges in Training PINNs: A Loss Landscape Perspective".
    Proceedings of the 41st International Conference on Machine
    Learning (ICML 2024). PMLR 235.

Verified citation (CrossRef-confirmed metadata for the closest baseline):
    Krishnapriyan, A. S., Gholami, A., Zhe, S., Kirby, R., and Mahoney,
    M. W. (2021). "Characterizing possible failure modes in physics-
    informed neural networks". NeurIPS 2021. arXiv:2109.01050.

Architecture details (Section 2.2 of paper):
    - Multilayer perceptron (MLP)
    - 3 hidden layers
    - Hidden widths: {50, 100, 200, 400}; addendum: width=200 worked best
    - tanh activations
    - Xavier normal initialization (Glorot & Bengio 2010); biases zero.
    - Input dim = 2 for (x, t) PDE problems studied (convection / reaction
      / wave); output dim = 1.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Plain multilayer perceptron with tanh activations.

    Mirrors the architecture used by Rathore et al. (2024) Section 2.2:
    three hidden layers, configurable width, tanh activations, Xavier
    normal init for weights, zero init for biases.
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden_widths: Sequence[int] = (200, 200, 200),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        widths = [in_dim, *hidden_widths, out_dim]
        layers = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
        self.layers = nn.ModuleList(layers)
        self.activation = self._get_activation(activation)
        self.reset_parameters()

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        if name == "tanh":
            return nn.Tanh()
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "sin":
            return _Sin()
        raise ValueError(f"Unknown activation: {name}")

    def reset_parameters(self) -> None:
        # Xavier normal for weights; zero for biases — Glorot & Bengio 2010.
        for m in self.layers:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h


class _Sin(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class PINN(nn.Module):
    """Physics-informed neural network wrapper.

    The PINN parameterizes the PDE solution u(x; w) as an MLP. The loss
    function (Eq. 2 in the paper) is computed in train.py via
    pinn_loss(...) using PDE-specific residual computations from
    the `pdes` module.

    Forward signature:
        u = pinn(coords)
    where coords is shape (N, in_dim).
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden_widths: Sequence[int] = (200, 200, 200),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.net = MLP(in_dim, out_dim, hidden_widths, activation)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
