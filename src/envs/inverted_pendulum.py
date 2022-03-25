"""
Inverted-Pendulum-v2 environment.
"""

from acme import specs

import chex
import dm_env
import gym
import numpy as np

# Inverted_Pendulum environment
class InvertedPendulumEnv(dm_env.Environment):
  def __init__(self, for_evaluation: bool) -> None:
    self._env = gym.make('InvertedPendulum-v2')
    self._for_evaluation = for_evaluation
    if self._for_evaluation:
      self.screens = []

  def step(self, action: chex.ArrayNumpy) -> dm_env.TimeStep:
    new_obs, reward, done, _ = self._env.step(action)
    if self._for_evaluation:
      self.screens.append(self._env.render(mode='rgb_array'))
    if done:
      return dm_env.termination(reward, new_obs)
    return dm_env.transition(reward, new_obs)

  def reset(self) -> dm_env.TimeStep:
    obs = self._env.reset()
    if self._for_evaluation:
      self.screens.append(self._env.render(mode='rgb_array'))
    return dm_env.restart(obs)

  def observation_spec(self) -> specs.BoundedArray:
    return specs.BoundedArray(shape=(4,), minimum=-float("inf"), maximum=float("inf"), dtype=np.float64)

  def action_spec(self) -> specs.BoundedArray:
    return specs.BoundedArray(shape=(1,), minimum=-3., maximum=3., dtype=np.float32)

  def close(self) -> None:
    self._env.close()