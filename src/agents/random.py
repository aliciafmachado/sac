"""
Random agent for baseline and testing.
"""

from agent import Agent
from acme import specs, types
import chex
from typing import *
from ..utils.trajectory import Trajectory

class RandomAgent(Agent):
  def __init__(self, environment_spec: specs.EnvironmentSpec) -> None:
    # TODO: implement
    pass

  def batched_actor_step(self, observation: types.NestedArray) -> types.NestedArray:
    # TODO: implement
    pass

  def learner_step(self, trajectory: Trajectory) -> Mapping[str, chex.ArrayNumpy]:
    # TODO: implement
    pass