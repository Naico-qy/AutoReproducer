"""SEMA model package.

Implements:
  * Functional adapters (AdaptFormer/LoRA/Convpass variants) -- Eq. 1
  * Representation descriptors (AE, LeakyReLU, latent=128) -- Eq. 2 + addendum
  * Expandable weighting routers -- Eq. 3
  * Self-expansion strategy -- Sec. 3.6
"""

from .adapters import FunctionalAdapter, LoRAAdapter, ConvPassAdapter, build_adapter
from .descriptor import RepresentationDescriptor
from .router import ExpandableRouter
from .modular_block import ModularAdapterBlock
from .architecture import SEMA

__all__ = [
    "FunctionalAdapter",
    "LoRAAdapter",
    "ConvPassAdapter",
    "build_adapter",
    "RepresentationDescriptor",
    "ExpandableRouter",
    "ModularAdapterBlock",
    "SEMA",
]
