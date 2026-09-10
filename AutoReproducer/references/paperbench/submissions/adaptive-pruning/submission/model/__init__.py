"""APT model package.

Implements the components of:
    Zhao, Hajishirzi & Cao,
    "APT: Adaptive Pruning and Tuning Pretrained Language Models for
     Efficient Training and Inference", ICML 2024.
"""

from .apt_adapter import APTAdapter, APTLinear
from .salience import (
    parameter_salience,
    outlier_aware_salience,
    adapter_salience,
    EMASalience,
)
from .pruning import (
    PruneController,
    binary_search_masks,
    cubic_sparsity_schedule,
)
from .tuning import RankController
from .distillation import SelfDistiller, layer_mapping, cofi_distill_loss
from .architecture import APTModel, build_apt_model

__all__ = [
    "APTAdapter",
    "APTLinear",
    "parameter_salience",
    "outlier_aware_salience",
    "adapter_salience",
    "EMASalience",
    "PruneController",
    "binary_search_masks",
    "cubic_sparsity_schedule",
    "RankController",
    "SelfDistiller",
    "layer_mapping",
    "cofi_distill_loss",
    "APTModel",
    "build_apt_model",
]
