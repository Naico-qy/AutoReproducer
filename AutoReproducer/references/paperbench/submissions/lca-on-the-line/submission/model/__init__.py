"""Model module for LCA-on-the-Line.

This package implements:
  * `lca`: Lowest Common Ancestor distance over WordNet / latent hierarchies
            (paper §2, §D.2, Appendix E.1, addendum: information-content score)
  * `architecture`: model wrappers around torchvision (VMs) and CLIP/OpenCLIP
            (VLMs); §A of the paper specifies all 75 backbones used
  * `losses`: LCA soft-label loss (Algorithm 1 in Appendix E.2)
  * `predictors`: OOD-error predictors (ID Top1, AC, Aline-D, Aline-S, ID LCA)
            from Table 3
"""

from .lca import (
    Hierarchy,
    WordNetHierarchy,
    KMeansLatentHierarchy,
    lca_distance,
    lca_distance_dataset,
    expected_lca_distance,
    build_lca_matrix,
    process_lca_matrix,
)
from .architecture import build_vm, build_vlm, ZeroShotClassifier
from .losses import LCAAlignmentLoss, weight_interpolate
from .predictors import (
    AccuracyOnTheLine,
    AverageConfidence,
    AlineD,
    AlineS,
    LCAPredictor,
)

__all__ = [
    "Hierarchy",
    "WordNetHierarchy",
    "KMeansLatentHierarchy",
    "lca_distance",
    "lca_distance_dataset",
    "expected_lca_distance",
    "build_lca_matrix",
    "process_lca_matrix",
    "build_vm",
    "build_vlm",
    "ZeroShotClassifier",
    "LCAAlignmentLoss",
    "weight_interpolate",
    "AccuracyOnTheLine",
    "AverageConfidence",
    "AlineD",
    "AlineS",
    "LCAPredictor",
]
