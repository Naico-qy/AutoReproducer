"""Network architectures used in the LBCS paper.

The paper specifies (Appendix D.2 / Table 7):
- F-MNIST: LeNet (LeCun et al., 1998) for both inner-loop proxy and final target.
- SVHN: a small CNN for the inner loop and a deeper CNN for the final model.
- CIFAR-10: a CNN for the inner loop and ResNet-18 (He et al., 2016) for the final model.
- MNIST-S (§5.1, addendum): ConvNet from Zhou et al. 2022 (Probabilistic Bilevel
  Coreset Selection,  https://github.com/x-zho14/Probabilistic-Bilevel-Coreset-Selection/blob/master/models.py).

Verified citation (CrossRef-friendly metadata, see ref_verify in submission tooling):
    Zhou, X., Pi, R., Zhang, W., Lin, Y., Chen, Z., & Zhang, T. (2022).
    "Probabilistic Bilevel Coreset Selection." ICML 2022, pp. 27287-27302.
    https://proceedings.mlr.press/v162/zhou22h.html
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# ----------------------- LeNet (F-MNIST) -----------------------
class LeNet(nn.Module):
    """Classic LeNet-5 (LeCun et al., 1998) used for F-MNIST in §5.2.

    Input is 1-channel 28x28 grayscale; we mirror the canonical 6-16-120-84-10
    structure used in (Borsos et al., 2020) and (Zhou et al., 2022).
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ---------- ConvNet from Zhou et al., 2022 (used in Figure 1 / §5.1) ----------
class ConvNet(nn.Module):
    """Two-block CNN (conv-dropout-maxpool-relu) from Zhou et al. 2022 (models.py).

    The addendum specifies that §5.1 (Table 1) and Figure 1 use exactly this
    ConvNet from Probabilistic Bilevel Coreset Selection (Zhou et al., 2022).
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout2d(0.25)
        # input 28x28 -> after two 2x2 max-pools => 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.dropout1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.dropout2(self.conv2(x)), 2))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Alias used in §5.1 / Figure 1
BorsosConvNet = ConvNet


# ----------------------- CNN for SVHN (Inner loop) -----------------------
# Per Appendix D.2 Table 7:
#   3x3 conv-relu, 3x3 conv-relu, 2x2 max-pool,
#   3x3 conv-relu, 2x2 max-pool,
#   Dense 8192 -> 1024 -> 256 -> 10.
class ConvNetSVHN(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 32 -> 16
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)  # 16 -> 8
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ----------------------- CNN for SVHN (trained-on-coresets) -----------------------
# Per Appendix D.2 Table 7 column 2 (deeper variant for final training).
class ConvNetSVHNTarget(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        # after three 2x2 maxpools: 32 -> 16 -> 8 -> 4
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ----------------------- CNN for CIFAR-10 (Inner loop) -----------------------
# Per Appendix D.2 Table 7 column 3 (5x5 conv block, then three 3x3 blocks).
class ConvNetCIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        # 32 -> 16 -> 8 -> 4 -> 2
        self.fc1 = nn.Linear(128 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def _resnet18(num_classes: int = 10, in_channels: int = 3) -> nn.Module:
    """ResNet-18 with the first conv adapted for 32x32 inputs (CIFAR-10 §5.2)."""
    m = resnet18(weights=None, num_classes=num_classes)
    if in_channels != 3 or True:  # always reshape for small inputs
        m.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        m.maxpool = nn.Identity()
    return m


def build_model(
    arch: str, num_classes: int = 10, in_channels: int | None = None
) -> nn.Module:
    """Factory that returns a network by name."""
    arch = arch.lower()
    if arch == "lenet":
        return LeNet(num_classes=num_classes, in_channels=in_channels or 1)
    if arch == "convnet":
        return ConvNet(num_classes=num_classes, in_channels=in_channels or 1)
    if arch == "convnet_svhn":
        return ConvNetSVHN(num_classes=num_classes, in_channels=in_channels or 3)
    if arch == "convnet_svhn_target":
        return ConvNetSVHNTarget(num_classes=num_classes, in_channels=in_channels or 3)
    if arch == "convnet_cifar":
        return ConvNetCIFAR(num_classes=num_classes, in_channels=in_channels or 3)
    if arch == "resnet18":
        return _resnet18(num_classes=num_classes, in_channels=in_channels or 3)
    raise ValueError(f"unknown arch '{arch}'")
