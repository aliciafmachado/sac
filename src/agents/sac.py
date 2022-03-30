"""
SAC agent.
"""

from os import environ
from src.agents.base_agent import Agent
from acme import specs, types
import chex
from typing import *
import jax
from jax import numpy as jnp
import haiku as hk
from ml_collections import ConfigDict
import optax
import rlax
from typing import *
from src.replay_buffers.buffer import ReplayBuffer
from src.utils.training_utils import LearnerState, ParamState, OptState, Transitions
from src.agents.networks import ValueNetwork, SoftQNetwork, PolicyNetwork


class SAC:
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
        # Save environment and config
        self.environment_spec = environment_spec
        self.config = config

        # Create RNG
        self._rng = jax.random.PRNGKey(seed=self.config.seed)

        # Create apply and init for neural networks
        self._init_policy, self._apply_policy = hk.without_apply_rng(hk.transform(self._hk_apply_policy))
        self._init_value, self._apply_value = hk.without_apply_rng(hk.transform(self._hk_apply_value))
        self._init_q, self._apply_q = hk.without_apply_rng(hk.transform(self._hk_apply_q))

        # Loss functions and their grads
        # We use value_and_grad so that we can log the loss
        self._grad_q1 = jax.value_and_grad(self._loss_fn_q, argnums=0)
        self._grad_q2 = jax.value_and_grad(self._loss_fn_q, argnums=1)
        self._grad_v = jax.value_and_grad(self._loss_fn_v)
        self._grad_pi = jax.value_and_grad(self._loss_fn_pi)
        
        # Jit functions
        self.init_fn = jax.jit(self._init_fn)
        self.update_fn = jax.jit(self._update_fn)
        self.apply_policy = jax.jit(self._apply_policy)
        self.apply_value = jax.jit(self._apply_value)
        self.apply_q = jax.jit(self._apply_q)
        self.get_action = jax.jit(self._get_action, static_argnums=3)

        # Create optimizers
        self.optimizer_q = optax.adam(self.config.q_lr)
        self.optimizer_v = optax.adam(self.config.v_lr)
        self.optimizer_p = optax.adam(self.config.p_lr)

        # Create replay buffer
        self.action_dim = environment_spec.actions.shape[-1]
        self.observation_dim = environment_spec.observations.shape[-1]
        self.buffer = ReplayBuffer(size_=self.config.replay_buffer_capacity,
                                   featuredim_=self.observation_dim, 
                                   actiondim_=self.action_dim)

        # Call initialization of learner state in the train loop
        # self._rng, key = jax.random.split(self._rng, 2)
        # learner_state = self.initialize()
        # q_t_target = jax.lax.stop_gradient(q_t_target)
        # ...
        # update_fn(learner_state, learner_state_target, transitions)
        # ...
        
    def initialize(self):
      self._rng, key = jax.random.split(self._rng, 2)
      # Create dummy inputs to initialize neural networks
      dummy_obs = jnp.expand_dims(jnp.zeros(self.environment_spec.observations.shape), axis=0)
      dummy_actions = jnp.expand_dims(jnp.zeros(self.environment_spec.actions.shape), axis=0)
      return self.init_fn(key, dummy_obs, dummy_actions)

    def _init_fn(self, rng: chex.PRNGKey, dummy_obs: types.NestedArray, 
                 dummy_actions: types.NestedArray) -> LearnerState:
        """
        Returns initial params and optimizer states provided dummy observation and action.
        """
        key_q1, key_q2, key_value, key_policy, key_target_value = jax.random.split(rng, 5)

        # Creating parameters
        q1_params = self._init_q(key_q1, dummy_obs, dummy_actions)
        q2_params = self._init_q(key_q2, dummy_obs, dummy_actions)
        v_params = self._init_value(key_value, dummy_obs)

        policy_params = self._init_policy(key_policy, dummy_obs)
        v_target_params = self._init_value(key_target_value, dummy_obs)

        # Creating opt states
        q1_opt_state = self.optimizer_q.init(q1_params)
        q2_opt_state = self.optimizer_q.init(q2_params)
        v_opt_state = self.optimizer_v.init(v_params)
        policy_opt_state = self.optimizer_p.init(policy_params)


        return LearnerState(
            params=ParamState(q1=q1_params, q2=q2_params, 
                              v=v_params, policy=policy_params,
                              v_target=v_target_params),
            opt_state=OptState(q1=q1_opt_state, q2=q2_opt_state, v=v_opt_state, policy=policy_opt_state),
        )

    def _loss_fn_q(self, q1_params: types.NestedArray,
                         q2_params: types.NestedArray,
                         v_target: types.NestedArray,
                         transitions: Transitions) -> chex.ArrayNumpy:
        """
        Loss function for Q networks.
        """
        # Calculate target value using target nn
        target_value    = self.apply_value(v_target, transitions.next_observations)
        target_q_value  = jax.lax.stop_gradient(target_value)
        target_q_value = (1. - transitions.dones[..., None]) * target_q_value # TODO
        target_q_value  = transitions.rewards + self.config.gamma * target_value

        # Calculate predicted q values using current nn
        predicted_q1_value = self.apply_q(q1_params, transitions.observations,
                                                           transitions.actions)
        predicted_q2_value = self.apply_q(q2_params, transitions.observations,
                                                           transitions.actions)

        # TD error
        td_error_q1 = target_q_value - predicted_q1_value
        td_error_q2 = target_q_value - predicted_q2_value

        # Loss for q1 and q2
        q1_loss = 0.5 * jnp.mean(jnp.square(td_error_q1))
        q2_loss = 0.5 * jnp.mean(jnp.square(td_error_q2))

        return q1_loss + q2_loss

    def _loss_fn_pi(self, policy_params: types.NestedArray, 
                         q1_params: types.NestedArray,
                         q2_params: types.NestedArray,
                         transitions: Transitions) -> chex.ArrayNumpy:
        """
        Loss function for policy network.
        """
        mu, sigma = self.apply_policy(policy_params, transitions.observations)
        self._rng, key = jax.random.split(self._rng, 2)
        new_actions = self._sample_action(key, mu, sigma)

        # We get the predicted new q value
        q1_pi = self.apply_q(q1_params, transitions.observations, new_actions)
        q2_pi = self.apply_q(q2_params, transitions.observations, new_actions)
        predicted_new_q_value = jax.lax.min(q1_pi, q2_pi)
        action_log_probs = self._get_logprob(new_actions, mu, sigma)

        policy_loss = jnp.mean(action_log_probs - predicted_new_q_value)
        
        # Adding regularization
        # l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(policy_params))

        return policy_loss # + 1e-2 * l2_loss

    def _loss_fn_v(self, v_params: types.NestedArray, 
                         policy_params: types.NestedArray,
                         q1_params: types.NestedArray,
                         q2_params: types.NestedArray,
                         transitions: Transitions) -> chex.ArrayNumpy:
        """
        Loss function for value network.
        """
        # Predict mu and sigma
        mu, sigma = self.apply_policy(policy_params, transitions.observations)

        # Split random number generator
        self._rng, key = jax.random.split(self._rng, 2)

        # Apply policy
        new_actions = self._sample_action(key, mu, sigma)
        action_log_probs = self._get_logprob(new_actions, mu, sigma)

        # Calculate predicted q value
        q1_pi = self.apply_q(q1_params, transitions.observations, new_actions)
        q2_pi = self.apply_q(q2_params, transitions.observations, new_actions)

        # Element wise minimum for stability
        q_pi = jax.lax.min(q1_pi, q2_pi)
        target_value_func = q_pi - action_log_probs
        target_value_func = jax.lax.stop_gradient(target_value_func)

        predicted_value = self.apply_value(v_params, transitions.next_observations)

        error = predicted_value - target_value_func

        # Compute the L2 error in expectation
        loss = 0.5 * jnp.square(error)
        loss = jnp.mean(loss)

        return loss

    def _update_fn(self, curr_ls: LearnerState,
                         transitions: Transitions) -> Tuple[LearnerState, Dict[str, float]]:
        """
        Update function.
        """
        ### Q network update
        loss_q1, grad_q1 = self._grad_q1(curr_ls.params.q1, curr_ls.params.q2, curr_ls.params.v_target, transitions)
        _, grad_q2 = self._grad_q2(curr_ls.params.q1, curr_ls.params.q2, curr_ls.params.v_target, transitions)

        # Apply gradients TODO check if this way is okay
        updates, curr_ls.opt_state.q1 = self.optimizer_q.update(grad_q1, curr_ls.opt_state.q1)
        curr_ls.params.q1 = optax.apply_updates(curr_ls.params.q1, updates)

        updates, curr_ls.opt_state.q2 = self.optimizer_q.update(grad_q2, curr_ls.opt_state.q2)
        curr_ls.params.q2 = optax.apply_updates(curr_ls.params.q2, updates)

        ### Policy network update
        loss_pi, grad_pi = self._grad_pi(curr_ls.params.policy, 
                                    curr_ls.params.q1, 
                                    curr_ls.params.q2,
                                    transitions)

        # Apply gradients
        updates, curr_ls.opt_state.policy = self.optimizer_q.update(grad_pi, 
                                                            curr_ls.opt_state.policy)
        curr_ls.params.policy = optax.apply_updates(curr_ls.params.policy, updates)

        ### Value network update
        loss_v, grad_v = self._grad_v(curr_ls.params.v, curr_ls.params.policy, 
                                      curr_ls.params.q1, curr_ls.params.q2, transitions)

        # Apply gradients
        updates, curr_ls.opt_state.v = self.optimizer_q.update(grad_v, 
                                                            curr_ls.opt_state.v)
        curr_ls.params.v = optax.apply_updates(curr_ls.params.v, updates)

        ### Target network update with polyak averaging
        curr_ls.params.v_target = jax.tree_multimap(lambda x, y: x + self.config.tau * (y - x),
                                    curr_ls.params.v_target, curr_ls.params.v)
        
        # Logging losses
        logs = {
          "loss_q": loss_q1,
          'loss_pi': loss_pi,
          "loss_v": loss_v,
        }

        return curr_ls, logs

    def _hk_apply_policy(self, observations: types.NestedArray) -> types.NestedArray:
        return PolicyNetwork([256, 256], self.environment_spec.actions)(observations)

    def _hk_apply_q(self, observations: types.NestedArray, actions: types.NestedArray) -> types.NestedArray:
        return SoftQNetwork([256, 256])(observations, actions)
        
    def _hk_apply_value(self, observations: types.NestedArray) -> types.NestedArray:
        return ValueNetwork([256, 256])(observations)

    def _get_action(self, key: chex.ArrayNumpy,
                          policy_params: types.NestedArray, 
                          observations: types.NestedArray,
                          deterministic=False) -> types.NestedArray:
      mu, sigma = self.apply_policy(policy_params, observations)
      if deterministic:
        return mu
      return self._sample_action(key, mu, sigma)

    def _sample_action(self, key: chex.ArrayNumpy,
                             mu: types.NestedArray, 
                             sigma: types.NestedArray) -> types.NestedArray:
      return rlax.squashed_gaussian().sample(key, mu, sigma, self.environment_spec.actions)

    def _get_logprob(self, actions: types.NestedArray, mu: types.NestedArray, 
                     sigma: types.NestedArray) -> types.NestedArray:
      return rlax.squashed_gaussian().logprob(actions, mu, sigma, self.environment_spec.actions)
