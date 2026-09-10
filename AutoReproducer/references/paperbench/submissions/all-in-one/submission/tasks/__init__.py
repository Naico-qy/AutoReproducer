"""Benchmark tasks for the Simformer (paper §4.1, addendum.md "Tasks").

Each task module exposes:
    prior()                 -> sample θ ~ p(θ)
    simulator(theta)        -> sample x ~ p(x|θ)
    num_params, num_data    -> dimensionalities
    structured_mask()       -> directed adjacency M_E for the joint
                               (theta first, x second).
"""

from .gaussian_linear import GaussianLinearTask
from .two_moons import TwoMoonsTask
from .slcp import SLCPTask
from .gaussian_mixture import GaussianMixtureTask
from .hmm import HMMTask
from .tree import TreeTask


TASK_REGISTRY = {
    "gaussian_linear": GaussianLinearTask,
    "two_moons": TwoMoonsTask,
    "slcp": SLCPTask,
    "gaussian_mixture": GaussianMixtureTask,
    "hmm": HMMTask,
    "tree": TreeTask,
}


def get_task(name: str):
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{name}'. Available: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[name]()
