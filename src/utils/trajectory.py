"""
Trajectory data class to save trajectories.
"""

from acme import types
import chex


@chex.dataclass
class Trajectory:
  observations: types.NestedArray  # [T, B, ...]
  actions: types.NestedArray  # [T, B, ...]
  rewards: chex.ArrayNumpy  # [T, B]
  dones: chex.ArrayNumpy  # [T, B]
  discounts: chex.ArrayNumpy # [T, B]