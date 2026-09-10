"""RL algorithm implementations used by CompoNet.

Provides:
  - SAC (Haarnoja et al. 2018) -- used for Meta-World (continuous actions).
  - PPO (Schulman et al. 2017) -- used for SpaceInvaders / Freeway (discrete).
  - CRL metrics (Section 5.1): success rate, average performance,
    forward transfer, RT, forgetting.

Both algorithms are based on CleanRL [Huang et al. JMLR 2022] which is
the implementation cited by the paper (Appendix E).
"""

from .sac import SAC, SACConfig
from .ppo import PPO, PPOConfig
from .metrics import (
    average_performance,
    forward_transfer,
    reference_transfer,
    forgetting,
    success_rate_curve,
)

__all__ = [
    "SAC",
    "SACConfig",
    "PPO",
    "PPOConfig",
    "average_performance",
    "forward_transfer",
    "reference_transfer",
    "forgetting",
    "success_rate_curve",
]
