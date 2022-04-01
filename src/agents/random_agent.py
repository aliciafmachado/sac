"""
Random agent for baseline and testing.
"""

from acme import specs, types
import chex
from typing import *
import jax
from jax import numpy as jnp
from src.utils.training_utils import LearnerState, Transitions


class RandomAgent :
  def __init__(self, environment_spec: specs.EnvironmentSpec):
    """
    Initializes random agent.
    """
    self.action_spec = environment_spec.actions
    self.get_action = jax.jit(self._get_action)

  def _get_action(self, rng: chex.ArrayNumpy, observation: types.NestedArray) -> types.NestedArray:
    """
    Returns random actions in response to batch of observations.
    """
    self.rng, subkey = jax.random.split(self.key)
    batch_size = jnp.shape(observation)[0]
    return jax.random.uniform(subkey, (batch_size, *self.action_spec.shape))

  def _update_fn(self, curr_ls: LearnerState,
                         transitions: Transitions) -> Tuple[LearnerState, Dict[str, float]]:
    """
    Returns empty dictionary since this agent doesn't learn.
    """
    return curr_ls, {}