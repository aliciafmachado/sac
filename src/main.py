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
import tensorflow as tf
from absl import app
import jax
import acme


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
    rng = jax.random.PRNGKey(0)
    environment = environments[config.env_idx]
    env = environment(for_evaluation=False)
    environment_spec = acme.make_environment_spec(env)

    model = SAC(environment_spec, config)

    # Call training of SAC agent
    all_returns, all_logs, num_total_steps, learner_state = train( environment = env,
                      agent = model,
                      rng = rng,
                      num_episodes=5,
                      num_steps=1,
                      min_buffer_capacity=50,
                      number_updates=1,
                      batch_size=10,
                      nb_updated_transitions=1
                      )


    print('done')

if __name__ == '__main__':
  app.run(main)