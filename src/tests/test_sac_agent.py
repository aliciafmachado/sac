"""
Unit test for SAC agent.
"""

from absl import flags
from ml_collections import config_flags
from src.train_agent import train
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


def __main__(argv):
    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], 'GPU')
    config = FLAGS.config
    environment = environments[config.env_idx]
    env = environment(for_evaluation=False)
    environment_spec = acme.make_environment_spec(env)
    model = SAC(environment_spec, config)
    # TODO: remove this rng being used by
    # implementing internal function to get action from model
    # TODO: add evaluation on the training loop
    rng = jax.random.PRNGKey(0)

    # Call training of SAC agent
    all_returns, all_logs, num_total_steps, learner_state = train( environment = env,
                      agent = model,
                      rng = rng,
                      num_episodes=20000,
                      num_steps=None,
                      min_buffer_capacity=config.min_buffer_capacity,
                      number_updates=8,
                      batch_size=config.batch_size,
                      nb_updated_transitions=8,
                      exploratory_policy_steps=config.exp_policy_steps,
                      )


    print('done')

    

if __name__ == '__main__':
  app.run(__main__)