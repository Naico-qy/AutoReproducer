"""LBCS model module: proxy networks and final-training networks.

References (per Appendix D.2 / Table 7 of the paper):
- LeNet for F-MNIST (proxy & target).
- ConvNet for SVHN (separate proxy / target variants).
- ConvNet for CIFAR-10 (proxy) and ResNet-18 (target).
- ConvNet for MNIST-S used in §5.1 (Borsos et al. 2020 / Zhou et al. 2022).
"""

from .architecture import (
    LeNet,
    ConvNet,
    ConvNetSVHN,
    ConvNetSVHNTarget,
    ConvNetCIFAR,
    BorsosConvNet,
    build_model,
)

__all__ = [
    "LeNet",
    "ConvNet",
    "ConvNetSVHN",
    "ConvNetSVHNTarget",
    "ConvNetCIFAR",
    "BorsosConvNet",
    "build_model",
]
