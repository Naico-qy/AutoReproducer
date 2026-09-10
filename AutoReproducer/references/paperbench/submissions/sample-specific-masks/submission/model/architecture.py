"""
Main SMM (Sample-specific Multi-channel Masks) model class.

Implements Eq. (3) from Cai et al., ICML 2024:
    f_in(x_i | phi, delta) = r(x_i) + delta ⊙ f_mask(r(x_i) | phi)

The pre-trained classifier f_P (and the output mapping f_out) are kept
frozen at training time -- only `delta` (shared noise pattern) and `phi`
(parameters of the mask generator) are updated, exactly as in Algorithm 1.

Reference (verified via CrossRef DOI 10.1109/CVPR52729.2023.01834):
    Chen, A., Yao, Y., Chen, P.-Y., Zhang, Y., & Liu, S.
    "Understanding and Improving Visual Prompting: A Label-Mapping Perspective."
    CVPR 2023. -- this is the ILM baseline we follow for the output mapping.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from .mask_generator import build_mask_generator, patch_wise_interpolation


# Per Appendix (addendum.md): ImageNet normalisation
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_pretrained(network: str) -> Tuple[nn.Module, int]:
    """Load and freeze an ImageNet-1k pre-trained classifier f_P.

    Returns:
        (model, input_size) -- input_size is 224 for ResNet, 384 for ViT_B32
        following the addendum specification.
    """
    network = network.lower()
    if network == "resnet18":
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        size = 224
    elif network == "resnet50":
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        size = 224
    elif network in ("vit_b32", "vit_b_32", "vitb32"):
        # ViT-B/32 with 384x384 inputs (per addendum).
        model = tvm.vit_b_32(weights=tvm.ViT_B_32_Weights.IMAGENET1K_V1)
        size = 384
    else:
        raise ValueError(f"Unknown network: {network}")
    # Freeze f_P
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, size


class SharedMask(nn.Module):
    """Pre-defined fixed binary mask for baseline VR methods (Section 5).

    - 'pad'    : padding-based reprogramming -- 1 outside center crop, 0 inside.
                 Center contains the resized target image; pattern lives in the
                 surrounding ring (Chen et al., 2023).
    - 'narrow' : watermark with a 1/8-width frame mask of 1s, rest 0.
    - 'medium' : watermark with a 1/4-width frame mask of 1s.
    - 'full'   : watermark over the entire image (M = all-ones).
    """

    def __init__(self, kind: str, image_size: int) -> None:
        super().__init__()
        H = W = image_size
        m = torch.zeros(1, 1, H, W)
        if kind == "full":
            m.fill_(1.0)
        elif kind == "pad":
            # Center kT-sized region is image, rest is pattern.
            inner = H // 2  # heuristic: image occupies the centre half
            start = (H - inner) // 2
            m.fill_(1.0)
            m[..., start : start + inner, start : start + inner] = 0.0
        elif kind == "narrow":
            border = max(H // 8, 1)  # width 28 for H=224
            m[..., :border, :] = 1.0
            m[..., -border:, :] = 1.0
            m[..., :, :border] = 1.0
            m[..., :, -border:] = 1.0
        elif kind == "medium":
            border = max(H // 4, 1)  # width 56 for H=224
            m[..., :border, :] = 1.0
            m[..., -border:, :] = 1.0
            m[..., :, :border] = 1.0
            m[..., :, -border:] = 1.0
        else:
            raise ValueError(f"Unknown shared-mask kind: {kind}")
        self.register_buffer("mask", m)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.mask.expand(batch_size, 3, -1, -1)


class SMM(nn.Module):
    """Sample-specific Multi-channel Masks framework.

    Implements Eq. (3):
        f_in(x_i | phi, delta) = r(x_i) + delta ⊙ f_mask(r(x_i) | phi)

    Training flow (Algorithm 1):
        1. r(x_i)  -- bilinear-interpolation upsample x_i to d_P (handled in
                      data/loader.py via transforms.Resize).
        2. m_i = f_mask(r(x_i) | phi) -> [B, 3, H/2^l, W/2^l] -> patch-wise
           interpolation up to [B, 3, H, W].
        3. f_in_i = r(x_i) + delta ⊙ m_i.
        4. logits = f_P(f_in_i)  -- f_P is frozen.
        5. logits_target = f_out(logits)  -- non-parametric label mapping.

    Args:
        network: backbone name -- 'resnet18' | 'resnet50' | 'vit_b32'.
        method: one of:
            'smm'                -> full method (Eq. 3, Sec 3.1)
            'only_delta'         -> r(x) + delta            (Sec 5 ablation)
            'only_fmask'         -> r(x) + f_mask(r(x))     (Sec 4, F^sp)
            'single_channel_smm' -> r(x) + delta * f_mask^s(r(x))
            'pad','narrow','medium','full' -> baseline shared-mask methods
        num_pool_layers: l in floor(H / 2^l). Defaults to 3 (patch_size=8).
    """

    def __init__(
        self, network: str = "resnet18", method: str = "smm", num_pool_layers: int = 3
    ) -> None:
        super().__init__()
        self.method = method
        self.network = network

        # f_P (frozen) and image size d_P
        self.classifier, self.image_size = load_pretrained(network)

        # ImageNet normalisation buffers (used for re-applying normalisation
        # after we add the noise pattern in the denormalised image space).
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        # Trainable shared noise pattern delta in R^{d_P}, initialised to zeros
        # (Algorithm 1). The pattern lives in the *normalised* image space.
        self.delta = nn.Parameter(torch.zeros(1, 3, self.image_size, self.image_size))

        # f_mask  (mask generator phi)
        self.num_pool_layers = num_pool_layers
        self.patch_size = 2**num_pool_layers

        if method in ("smm", "only_fmask"):
            self.fmask = build_mask_generator(network, num_pool_layers, out_channels=3)
            self.shared_mask = None
        elif method == "single_channel_smm":
            self.fmask = build_mask_generator(network, num_pool_layers, out_channels=1)
            self.shared_mask = None
        elif method == "only_delta":
            self.fmask = None
            self.shared_mask = SharedMask("full", self.image_size)
        elif method in ("pad", "narrow", "medium", "full"):
            self.fmask = None
            self.shared_mask = SharedMask(method, self.image_size)
        else:
            raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # Eq. (3) implementation
    # ------------------------------------------------------------------
    def f_in(self, r_x: torch.Tensor) -> torch.Tensor:
        """Apply input visual-reprogramming on resized images r(x).

        Args:
            r_x: [B, 3, H, W] -- already resized & ImageNet-normalised.

        Returns:
            f_in(x): [B, 3, H, W] -- reprogrammed input ready for f_P.
        """
        B = r_x.shape[0]
        delta = self.delta.expand(B, -1, -1, -1)

        if self.method == "only_delta":
            mask = self.shared_mask(B)
            return r_x + delta * mask

        if self.method in ("pad", "narrow", "medium", "full"):
            mask = self.shared_mask(B)
            return r_x + delta * mask

        if self.method == "only_fmask":
            # F^sp: r(x) + f_mask(r(x)) -- no shared delta.
            m = self.fmask(r_x)
            if self.fmask.out_channels == 1:
                m = m.expand(-1, 3, -1, -1)
            m = patch_wise_interpolation(m, self.patch_size)
            return r_x + m

        if self.method == "single_channel_smm":
            m = self.fmask(r_x)  # [B, 1, h, w]
            m = m.expand(-1, 3, -1, -1)  # broadcast to 3 channels
            m = patch_wise_interpolation(m, self.patch_size)
            return r_x + delta * m

        # Default: full SMM (Eq. 3)
        m = self.fmask(r_x)  # [B, 3, h, w]
        m = patch_wise_interpolation(m, self.patch_size)
        return r_x + delta * m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """End-to-end forward: x is already r(x) (resized + normalised).

        Returns ImageNet logits (length |Y^P| = 1000). Output mapping
        f_out is applied externally (utils.label_mapping).
        """
        x = self.f_in(x)
        with torch.set_grad_enabled(self.training):
            # We allow gradients to flow back through the frozen f_P to delta
            # and phi (its parameters are not updated because requires_grad=False).
            logits = self.classifier(x)
        return logits

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def trainable_parameters(self):
        """Return iterator over trainable params (delta + phi)."""
        params = [self.delta]
        if self.fmask is not None:
            params += list(self.fmask.parameters())
        return params

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())
