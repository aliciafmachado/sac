"""
Definition of util functions for training.
"""

import chex
from acme import types


@chex.dataclass
class ParamState:
  policy: types.NestedArray 
  v: types.NestedArray  
  q1: types.NestedArray  
  q2: types.NestedArray

@chex.dataclass
class OptState:
  policy: types.NestedArray 
  v: types.NestedArray  
  q1: types.NestedArray  
  q2: types.NestedArray

@chex.dataclass
class LearnerState:
  params: ParamState
  opt_state: OptState

@chex.dataclass
class Transitions:
  observations: types.NestedArray  # [T, B, ...]
  actions: types.NestedArray  # [T, B, ...]
  next_observations: types.NestedArray  # [T, B, ...]
  rewards: chex.ArrayNumpy  # [T, B]
  dones: chex.ArrayNumpy  # [T, B]