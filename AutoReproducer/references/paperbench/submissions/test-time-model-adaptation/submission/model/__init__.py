# Model package for FOA: Test-Time Forward-Optimization Adaptation
# Reference: Niu et al., "Test-Time Model Adaptation with Only Forward Passes", ICML 2024
from .architecture import PromptedViT, build_vit_base
from .foa import FOA, SourceStats
from .foa_interval import FOAInterval
from .activation_shift import ActivationShifter
from .baselines import TENT, T3A, NoAdapt
from .quantization import quantize_vit_8bit

__all__ = [
    "PromptedViT",
    "build_vit_base",
    "FOA",
    "FOAInterval",
    "SourceStats",
    "ActivationShifter",
    "TENT",
    "T3A",
    "NoAdapt",
    "quantize_vit_8bit",
]
