"""CompoNet model package.

Implements the Self-Composing Policies Network from
"Self-Composing Policies for Scalable Continual Reinforcement Learning"
(Malagon, Ceberio, Lozano, ICML 2024).
"""

from .architecture import (
    CompoNet,
    CompoNetActor,
    SelfComposingPolicyModule,
    OutputAttentionHead,
    InputAttentionHead,
    InternalPolicy,
    PositionalEncoding,
    AtariEncoder,
    ProgressiveNet,
    PackNet,
    BaselineActor,
)

__all__ = [
    "CompoNet",
    "CompoNetActor",
    "SelfComposingPolicyModule",
    "OutputAttentionHead",
    "InputAttentionHead",
    "InternalPolicy",
    "PositionalEncoding",
    "AtariEncoder",
    "ProgressiveNet",
    "PackNet",
    "BaselineActor",
]
