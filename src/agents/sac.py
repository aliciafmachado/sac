"""
SAC agent.
"""

from base_agent import Agent, Trajectory
from acme import specs, types
import chex
from typing import *
import jax
from jax import numpy as jnp
import haiku as hk


class MLP(hk.Module):
    def __init__():
        pass

    def __call__():
        pass


class PolicyNetwork(hk.Module):
    def __init__():
        pass

    def __call__():
        pass


class SAC(Agent):
    """
    Soft Actor Critic according to:

    Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). 
    Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning 
    with a Stochastic Actor. doi:10.48550/ARXIV.1801.01290

    https://arxiv.org/abs/1801.01290
    """
    def __init__(self, environment_spec: specs.EnvironmentSpec) -> None:
        # TODO: fix initialization
        self.action_spec = environment_spec.actions
        self.q1 = None
        self.q2 = None
        self.policy = None

    def loss_fn_q():
        # TODO
        pass

    def loss_fn_policy():
        # TODO
        pass

    def loss_fn_value():
        # TODO
        pass

    def update_fn(self, trajectory: Trajectory) -> Mapping[str, chex.ArrayNumpy]:
        """
        Returns empty dictionary since this agent doesn't learn.
        """
        # TODO
        return dict()

    def evaluate():
        pass
 
    # # The fn below is not useful since it's for an online policy
    # TODO: refactor base agent (abstract class)
    # def batched_actor_step(self, observation: types.NestedArray) -> types.NestedArray:
    #     """
    #     Returns random actions in response to batch of observations.
    #     """
    #     pass
