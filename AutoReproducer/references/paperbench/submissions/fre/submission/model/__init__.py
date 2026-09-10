"""FRE neural-network components.

Public exports:
    FREEncoder         -- permutation-invariant transformer over (s, eta(s)) pairs
    FREDecoder         -- MLP that predicts eta(s) from (s, z)
    FRE                -- joint encoder + decoder + ELBO loss
    Actor / Critic / VNet  -- IQL components conditioned on z
    FREAgent           -- training wrapper combining FRE + IQL

Verified reference:
    Kostrikov, Nair, Levine. "Offline Reinforcement Learning with Implicit
    Q-Learning." arXiv:2110.06169, 2021.
    (CrossRef does not index arXiv DOIs as of 2024 -- ref_verify failure
     is expected for arXiv-only references; metadata above was checked
     against Semantic Scholar / OpenAlex search results.)
"""

from .architecture import (
    FREEncoder,
    FREDecoder,
    FRE,
    Actor,
    Critic,
    VNet,
    FREAgent,
)

__all__ = [
    "FREEncoder",
    "FREDecoder",
    "FRE",
    "Actor",
    "Critic",
    "VNet",
    "FREAgent",
]
