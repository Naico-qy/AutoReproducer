"""Knowledge-retention auxiliary losses + RL trainers used in the paper.

This module is the algorithmic core of the paper. It implements:

* `aux_losses.bc_loss`  - Behavioral Cloning auxiliary loss (App. C.2 / Eq. 7).
* `aux_losses.ks_loss`  - Kickstarting auxiliary loss (App. C.2 / Eq. 8).
* `aux_losses.ewc_loss` - Elastic Weight Consolidation (App. C.1 / Eq. 6).
* `fisher.estimate_fisher_diagonal` - Fisher diagonal from 10 000 NLD-AA
  batches (Addendum).
* `episodic_memory.EpisodicMemory` - frozen 10 % slice of SAC's replay buffer.
* `appo.APPOTrainer` - asynchronous PPO for NetHack.
* `ppo_rnd.PPORNDTrainer` - PPO + Random Network Distillation for Montezuma.
* `sac.SACTrainer` - Soft Actor-Critic for RoboticSequence.
"""

from . import aux_losses, fisher, episodic_memory  # noqa: F401
from .appo import APPOTrainer
from .ppo_rnd import PPORNDTrainer
from .sac import SACTrainer

__all__ = ["APPOTrainer", "PPORNDTrainer", "SACTrainer"]
