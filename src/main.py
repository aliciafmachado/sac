"""
Main file for passing the parameters and calling training.
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

    # model._update_fn(ls, transitions)
    # Call training of SAC agent
    # Config example of usage on:
    # https://github.com/google/flax/blob/390383830bd2de784994d4d961e1ffc42a249962/examples/ppo/ppo_lib.py#L277
    train(model, env, config)


if __name__ == '__main__':
  app.run(main)