"""RICE model components.

Modules:
    architecture     -- Generic Actor-Critic MLP (paper §4.1: SB3 MlpPolicy)
    mask_network     -- Algorithm 1 of the paper (mask-net training)
    rnd              -- Random Network Distillation exploration bonus
    explanation      -- Critical-state identification using the mask net
    refiner          -- Algorithm 2 (RICE refinement)
    baselines        -- PPO-FT, StateMask-R, JSRL, Random-Explanation baselines
"""

from .architecture import ActorCritic, MaskNet
from .mask_network import MaskNetworkTrainer
from .rnd import RNDModule
from .explanation import identify_critical_state, fidelity_score
from .refiner import RICERefiner
from .baselines import PPOFinetune, StateMaskR, JSRL, RandomExplanation

__all__ = [
    "ActorCritic",
    "MaskNet",
    "MaskNetworkTrainer",
    "RNDModule",
    "identify_critical_state",
    "fidelity_score",
    "RICERefiner",
    "PPOFinetune",
    "StateMaskR",
    "JSRL",
    "RandomExplanation",
]
