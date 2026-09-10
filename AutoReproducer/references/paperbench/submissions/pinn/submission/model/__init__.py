"""PINN model module.

Implements multilayer perceptron architectures used in
Rathore et al. (2024), "Challenges in Training PINNs: A Loss
Landscape Perspective", ICML 2024.
"""

from .architecture import PINN, MLP

__all__ = ["PINN", "MLP"]
