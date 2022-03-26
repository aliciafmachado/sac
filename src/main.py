"""
Main file for passing the parameters and calling training.
"""

from absl import flags
from ml_collections import config_flags
from src.train_agent import train
from src.agents.sac import SAC
import tensorflow as tf
from absl import app


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    "configs/default.py",
    'File path to the default configuration file.',
    lock_config=True)


def main(argv):
    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], 'GPU')
    config = FLAGS.config
    model = SAC(config)
    # Call training of SAC agent
    # Config example of usage on:
    # https://github.com/google/flax/blob/390383830bd2de784994d4d961e1ffc42a249962/examples/ppo/ppo_lib.py#L277
    train(model, config)


if __name__ == '__main__':
  app.run(main)