"""
Abstract agent to build off of.
"""

# This code is not authoral: taken from TD3 and modified
import abc

from acme import types
import chex
from typing import Mapping
from src.utils.training_utils import Transitions


class Agent(abc.ABC):
    @abc.abstractmethod
    def learner_step(self, transitions: Transitions) -> Mapping[str, chex.ArrayNumpy]:
        """One step of learning on a trajectory.
        
        The mapping returned can contain various logs.
        """
        pass

    @abc.abstractmethod
    def batched_actor_step(self, observations: types.NestedArray) -> types.NestedArray:
        """Returns actions in response to observations.
        
        Observations are assumed to be batched, i.e. they are typically arrays, or
        nests (think nested dictionaries) of arrays with shape (B, F_1, F_2, ...)
        where B is the batch size, and F_1, F_2, ... are feature dimensions.
        """
        pass