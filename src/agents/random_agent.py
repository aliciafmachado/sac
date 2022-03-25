"""
Random agent for baseline and testing.
"""

from src.agents.base_agent import Agent, Trajectory
from acme import specs, types
import chex
from typing import *
import jax
from jax import numpy as jnp


class RandomAgent(Agent):
  def __init__(self, environment_spec: specs.EnvironmentSpec) -> None:
    # TODO: add random key as arg so that we can use jnp instead of np
    self.action_spec = environment_spec.actions
    self.seed = 0
    self.key = jax.random.PRNGKey(self.seed)

  def batched_actor_step(self, observation: types.NestedArray) -> types.NestedArray:
    """
    Returns random actions in response to batch of observations.
    """
    # TODO: maybe use vmap instead of batch size
    self.key, subkey = jax.random.split(self.key)
    batch_size = jnp.shape(observation)[0]
    return jax.random.uniform(subkey, (batch_size, *self.action_spec.shape))

  def learner_step(self, trajectory: Trajectory) -> Mapping[str, chex.ArrayNumpy]:
    """
    Returns empty dictionary since this agent doesn't learn.
    """
    return dict()