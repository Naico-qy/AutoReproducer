"""
Mask Generator f_mask for SMM (Sample-specific Multi-channel Masks).

Based on Section 3.2 + Appendix A.2 of:
    Cai, Ye, Feng, Qi, Liu. "Sample-specific Masks for Visual Reprogramming-based Prompting."
    ICML 2024.

Architecture (Appendix A.2, Figures 8 & 9):
- 5-layer CNN for ResNet backbones (input 224x224x3)
- 6-layer CNN for ViT_B32 backbone (input 384x384x3)
- All conv layers use 3x3 kernels with padding=1, stride=1 (preserves spatial size).
- 3 MaxPool 2x2 stride-2 layers reduce spatial size by 8x.
- Final layer outputs 3 channels (or 1 for the single-channel ablation).
- Output spatial size: floor(H / 2^l) x floor(W / 2^l), where l is the number
  of MaxPool layers (default l=3, patch_size=2^3=8 from Sec. 5).

Parameter counts (Table 4):
- 5-layer (ResNet): 26,499 params
- 6-layer (ViT)   : 102,339 params
"""

import torch
import torch.nn as nn


class MaskGenerator5Layer(nn.Module):
    """5-layer CNN mask generator for ResNet-18 / ResNet-50 backbones.

    Channel progression (Appendix A.2 / Figure 8):
        3 -> 8 -> 16 -> 32 -> 16 -> 3
    Three 2x2 MaxPool layers spread across the network give an 8x downscaling.
    """

    def __init__(
        self, in_channels: int = 3, out_channels: int = 3, num_pool_layers: int = 3
    ) -> None:
        super().__init__()
        self.num_pool_layers = num_pool_layers
        self.out_channels = out_channels

        # Five 3x3 conv layers, padding=1, stride=1 -> preserve H, W.
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1, stride=1)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()  # mask values in [0, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, 3, H, W]
        # Conv block 1
        h = self.relu(self.conv1(x))
        if self.num_pool_layers >= 1:
            h = self.pool(h)
        # Conv block 2
        h = self.relu(self.conv2(h))
        if self.num_pool_layers >= 2:
            h = self.pool(h)
        # Conv block 3
        h = self.relu(self.conv3(h))
        if self.num_pool_layers >= 3:
            h = self.pool(h)
        # Conv block 4
        h = self.relu(self.conv4(h))
        if self.num_pool_layers >= 4:
            h = self.pool(h)
        # Final 1x1-equivalent conv -> mask logits
        h = self.conv5(h)
        return self.sigmoid(h)


class MaskGenerator6Layer(nn.Module):
    """6-layer CNN mask generator for ViT_B32 (input 384x384x3).

    Channel progression (Appendix A.2 / Figure 9):
        3 -> 8 -> 16 -> 32 -> 32 -> 16 -> 3
    Three 2x2 MaxPool layers -> 8x downscaling.
    """

    def __init__(
        self, in_channels: int = 3, out_channels: int = 3, num_pool_layers: int = 3
    ) -> None:
        super().__init__()
        self.num_pool_layers = num_pool_layers
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1, stride=1)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.conv1(x))
        if self.num_pool_layers >= 1:
            h = self.pool(h)
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        if self.num_pool_layers >= 2:
            h = self.pool(h)
        h = self.relu(self.conv4(h))
        h = self.relu(self.conv5(h))
        if self.num_pool_layers >= 3:
            h = self.pool(h)
        h = self.conv6(h)
        return self.sigmoid(h)


def build_mask_generator(
    network: str, num_pool_layers: int = 3, out_channels: int = 3
) -> nn.Module:
    """Factory: 5-layer CNN for ResNet, 6-layer CNN for ViT_B32."""
    if network.lower().startswith("vit"):
        return MaskGenerator6Layer(
            in_channels=3, out_channels=out_channels, num_pool_layers=num_pool_layers
        )
    # Default: 5-layer CNN for ResNet-18 / ResNet-50.
    return MaskGenerator5Layer(
        in_channels=3, out_channels=out_channels, num_pool_layers=num_pool_layers
    )


def patch_wise_interpolation(mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Patch-wise interpolation (Section 3.3).

    Upscales each pixel of `mask` to a patch_size x patch_size block by
    nearest-neighbor replication. Avoids floating point arithmetic and has
    no gradient computation in back-prop (the operation is a pure
    rearrangement / repeat). This is the same as ``F.interpolate(..., mode='nearest')``
    with scale_factor=patch_size, which we use for efficiency.

    Args:
        mask: tensor of shape [B, C, h, w].
        patch_size: integer 2^l.

    Returns:
        Tensor of shape [B, C, h*patch_size, w*patch_size].
    """
    if patch_size <= 1:
        return mask
    # repeat_interleave is equivalent to copying the same value to a 2^l x 2^l
    # neighbourhood. It has zero gradient through the value-replication itself.
    return mask.repeat_interleave(patch_size, dim=2).repeat_interleave(
        patch_size, dim=3
    )
