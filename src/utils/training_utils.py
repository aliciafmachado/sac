"""
Definition of util functions for training.
"""

import chex
from acme import types


@chex.dataclass
class LearnerState:
  policy_params: types.NestedArray 
  v_params: types.NestedArray  
  q1_params: chex.ArrayNumpy  
  q2_params: chex.ArrayNumpy  
