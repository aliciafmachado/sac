"""
SAC agent.
"""

from os import environ
from src.agents.base_agent import Agent, Transitions
from acme import specs, types
import chex
from typing import *
import jax
from jax import numpy as jnp
import haiku as hk
from ml_collections import ConfigDict
import optax
from typing import *
from src.replay_buffers.buffer import ReplayBuffer
from src.utils.training_utils import LearnerState

# TODO: fix what the functions receive as input and what they return
# TODO: test functions and check if jit works properly in the class
# TODO: add done to replay buffer
# TODO: I think that the learnerstate should not be part of the class
# but i'm not entirely sure how to deal with the target network...


class ValueNetwork(hk.Module):
  def __init__(self, output_sizes: Sequence[int], name: Optional[str] = None) -> None:
    super().__init__(name=name)
    self._output_sizes = output_sizes

  def __call__(self, observations: chex.Array) -> chex.Array:
    h = observations

    for i, o in enumerate(self._output_sizes):
      h = hk.Linear(o)(h)
      h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)
      h = jax.nn.relu(h)
    return hk.Linear(1)(h)[..., 0]

class SoftQNetwork(hk.Module):
  def __init__(self, output_sizes: Sequence[int], name: Optional[str] = None) -> None:
    super().__init__(name=name)
    self._output_sizes = output_sizes

  def __call__(self, observations: chex.Array, actions: chex.Array) -> chex.Array:
    h = jnp.concatenate([observations, actions], axis=1)

    for i, o in enumerate(self._output_sizes):
      h = hk.Linear(o)(h)
      h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)
      h = jax.nn.relu(h)
    return hk.Linear(1)(h)[..., 0]

class PolicyNetwork(hk.Module):
  def __init__(self, output_sizes: Sequence[int], action_spec: specs.BoundedArray, name: Optional[str] = None) -> None:
    super().__init__(name=name)
    self._output_sizes = output_sizes
    self._action_spec = action_spec

  def __call__(self, x: chex.Array, ) -> Tuple[chex.Array, chex.Array]:
    action_shape = self._action_spec.shape
    action_dims = jnp.prod(action_shape)
    h = x
    for i, o in enumerate(self._output_sizes):
      h = hk.Linear(o)(h)
      h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)
      h = jax.nn.relu(h)
    h = hk.Linear(2 * action_dims)(h)
    mu, pre_sigma = jnp.split(h, 2, axis=-1)
    sigma = jax.nn.softplus(pre_sigma)
    return hk.Reshape(action_shape)(.1 * mu), hk.Reshape(action_shape)(.1 * sigma)


class SAC(Agent):
    """
    Soft Actor Critic according to:

    Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). 
    Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning 
    with a Stochastic Actor. doi:10.48550/ARXIV.1801.01290

    https://arxiv.org/abs/1801.01290
    """

    def __init__(self, environment_spec: specs.EnvironmentSpec, config: ConfigDict) -> None:
        """
        Initialize the agent.

        Args:
            environment_spec: environment specification
            config: configuration

        Returns:
            None
        """

        # TODO: fix initialization
        # TODO: Add target functions, and initialize them
        self.environment_spec = environment_spec
        self.config = config

        # Create RNG
        self._rng = jax.random.PRNGKey(seed=self.config.seed)

        # Create apply and init for neural networks
        self._init_policy, self._apply_policy = hk.without_apply_rng(hk.transform(self._hk_apply_policy))
        self._init_value, self._apply_value = hk.without_apply_rng(hk.transform(self._hk_apply_value))
        self._init_q, self._apply_q = hk.without_apply_rng(hk.transform(self._hk_apply_q))

        # TODO: check how the init_loss, apply_loss, and grad would work in this structure:
        # self._init_loss, apply_loss = hk.without_apply_rng(hk.transform(self._loss_function))

        # Jit functions
        self.init_fn = jax.jit(self._init_fn)
        self.update_fn = jax.jit(self._update_fn)
        self.apply_policy = jax.jit(self._apply_policy)
        self.apply_value = jax.jit(self._apply_value)
        self.apply_q1 = jax.jit(self._apply_q1)
        self.apply_q2 = jax.jit(self._apply_q2)

        # Create optimizers
        self.optimizer_q = optax.adam(self.config.learning_rate_q)
        self.optimizer_v = optax.adam(self.config.learning_rate_v)
        self.optimizer_p = optax.adam(self.config.learning_rate_p)
        
        # Create replay buffer
        self.buffer = ReplayBuffer(self.config.replay_buffer_capacity)

        # Initialize LearnerState
        # Create dummy inputs to initialize neural networks
        dummy_obs = jnp.expand_dims(jnp.zeros(self.environment_spec.observations.shape), axis=0)
        dummy_actions = jnp.expand_dims(jnp.zeros(self.environment_spec.actions.shape), axis=0)

        # Call initialization
        self._rng, key = jax.random.split(self._rng, 2)
        self.learner_state = self.init_fn(key, dummy_obs, dummy_actions)
        

    def _init_fn(self, rng: chex.PRNGKey, dummy_obs: types.NestedArray, 
                 dummy_actions: types.NestedArray) -> LearnerState:
        """
        Returns initial params and optimizer states provided dummy observation and action.
        """
        key_q1, key_q2, key_value, key_policy = jax.random.split(rng, 4)

        # TODO: do we add the optimizer state in Learner state as well?
        # Creating parameters
        q1_params = self._init_q(key_q1, dummy_obs, dummy_actions)
        q2_params = self._init_q(key_q2, dummy_obs, dummy_actions)
        v_params=self._init_value(key_value, dummy_obs)
        policy_params=self._init_policy(key_policy, dummy_obs)

        return LearnerState(
            q1_params=q1_params,
            q2_params=q2_params,
            v_params=v_params,
            policy_params=policy_params,
            q1_opt_state=self.optimizer_q.init(q1_params),
            q2_opt_state=self.optimizer_q.init(q2_params),
            v_opt_state=self.optimizer_v.init(v_params),
            policy_opt_state=self.optimizer_policy.init(policy_params),
        )

    def _loss_fn_q(self, learner_state: LearnerState, transitions: Transitions):
        """
        Loss function for Q networks.
        """

        # TODO
        pass

    def _loss_fn_pi(self, learner_state: LearnerState, transitions: Transitions):
        """
        Loss function for policy network.
        """

        # TODO
        pass

    def _loss_fn_v(self, learner_state: LearnerState, transitions: Transitions):
        """
        Loss function for value network.
        """

        # TODO: extract observation from transition
        # out = self.value(observations)
        # actions, log_pi = self.policy(observations)
        # log_target_1 = self.q1(observations, actions)
        # log_target_2 = self.q2(observations, actions)
        # min_log_target = jnp.minimum(log_target_1, log_target_2)
        # Reparameterize: todo ??

        pass

    def _update_fn(self, learner_state: LearnerState,
                         learner_state_target: LearnerState,
                         transitions: Transitions) -> Tuple[LearnerState, LogsDict]:
        """
        Update function.
        """
        # TODO: add target network argument to loss functions above
        # TODO: freeze other neural networks when updating a specific one
        # Q network update

        # Value network update

        # Policy network update

        # Target network update
        

        # TODO: add logs
        logs = dict()

        return learner_state, logs

    def _hk_apply_policy(self, observations: types.NestedArray) -> types.NestedArray:
        return PolicyNetwork([256, 256], self.environment_spec.actions)(observations)

    def _hk_apply_q(self, observations: types.NestedArray, actions: types.NestedArray) -> types.NestedArray:
        return SoftQNetwork([256, 256])(observations, actions)
        
    def _hk_apply_value(self, observations: types.NestedArray) -> types.NestedArray:
        return ValueNetwork([256, 256])(observations)

    def _batched_actor_step(self, observation: types.NestedArray) -> types.NestedArray:
        """
        Returns random actions in response to batch of observations.
        """
        pass
