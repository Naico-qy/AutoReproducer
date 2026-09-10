"""Neural network architectures used in the paper.

* `nethack_net.NetHackNet` - LSTM + ResNet actor-critic, the 30M Tuyls et al.
  (2023) checkpoint architecture (App. B.1).
* `montezuma_net.RNDNetwork`, `RNDFeatureNet` - PPO+RND networks (App. B.2).
* `sac_net.GaussianActor`, `QFunction` - SAC actor and Q-function used for
  RoboticSequence (App. B.3).
"""

from .nethack_net import NetHackNet
from .montezuma_net import MontezumaPolicy, RNDPredictor, RNDTarget
from .sac_net import GaussianActor, QFunction

__all__ = [
    "NetHackNet",
    "MontezumaPolicy",
    "RNDPredictor",
    "RNDTarget",
    "GaussianActor",
    "QFunction",
]
