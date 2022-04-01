"""
Definition of util functions for training.
"""

import chex
from acme import types
from src.envs.inverted_pendulum import InvertedPendulumEnv
from src.envs.pendulum import PendulumEnv
from src.envs.reacher import ReacherEnv


@chex.dataclass
class ParamState:
  policy: types.NestedArray 
  v: types.NestedArray  
  q1: types.NestedArray  
  q2: types.NestedArray
  v_target: types.NestedArray


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


environments = {
  0: PendulumEnv,
  1: InvertedPendulumEnv,
  2: ReacherEnv,
}


env_names = {
  0: 'PendulumEnv',
  1: 'InvertedPendulumEnv',
  2: 'ReacherEnv',
}
