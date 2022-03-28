"""
Test loss and update fns.
"""

from absl import flags
from ml_collections import config_flags
from src.train_agent import train
from src.agents.sac import SAC
from src.envs.inverted_pendulum import InvertedPendulumEnv
from src.envs.pendulum import PendulumEnv
from src.envs.reacher import ReacherEnv
from src.utils.training_utils import Transitions
import tensorflow as tf
from absl import app
import acme
from jax import numpy as jnp

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    "src/configs/default.py",
    'File path to the default configuration file.',
    lock_config=True)


environments = {
  0: PendulumEnv,
  1: InvertedPendulumEnv,
  2: ReacherEnv,
}


def main(argv):
    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], 'GPU')
    config = FLAGS.config
    environment = environments[config.env_idx]
    env = environment(for_evaluation=False)
    environment_spec = acme.make_environment_spec(env)
    model = SAC(environment_spec, config)
    ls = model.initialize()

    obs_shp = (1, *environment_spec.observations.shape)
    act_shp = (1, *environment_spec.actions.shape)

    fake_obs = jnp.concatenate([jnp.zeros(obs_shp), jnp.ones(obs_shp)], axis=0)
    fake_actions = jnp.concatenate([jnp.zeros(act_shp), jnp.ones(act_shp)], axis=0)
    fake_n_obs = jnp.concatenate([jnp.ones(obs_shp), jnp.zeros(obs_shp)], axis=0)
    fake_reward = jnp.full((2,1), 0.2)
    fake_dones = jnp.full((2,1), 0)

    print(fake_obs.shape)
    print(fake_actions.shape)
    print(fake_n_obs.shape)
    print(fake_reward.shape)
    print(fake_dones.shape)


    transitions = Transitions(
      observations=fake_obs,
      actions=fake_actions,
      next_observations=fake_n_obs,
      rewards=fake_reward,
      dones=fake_dones,
    )

    # print(model._loss_fn_q(ls.params, transitions).item())

    # mu, sigma = model.apply_policy(ls.params.policy, fake_obs)

    # print(model._loss_fn_v(mu, sigma, ls.params, transitions).item())
    # print(model._loss_fn_pi(mu, sigma, ls.params, transitions).item())

    ls, logs = model.update_fn(ls, transitions)
    print(logs)
    ls, logs = model.update_fn(ls, transitions)
    print(logs)
    ls, logs = model.update_fn(ls, transitions)
    print(logs)


if __name__ == '__main__':
  app.run(main)