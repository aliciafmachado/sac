"""
Unit test for SAC agent.
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


def __main__():
    n_simulations = 5
    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], 'GPU')
    config = FLAGS.config
    environment = environments[config.env_idx]
    env = environment(for_evaluation=False)
    environment_spec = acme.make_environment_spec(env)
    model = SAC(environment_spec, config)

    # TODO: finish this test
    # for _ in range(n_simulations):
    #     ts = env.reset()
    #     while True:
    #         batched_observation = tree.map_structure(lambda x: x[None], ts.observation)
    #         a = random_agent.batched_actor_step(batched_observation)[0]
    #         ts = env.step(a)
    #         if render:
    #             env._env.render()
    #         if ts.last():
    #             break

    # env.close()
    # print("Done.")

if __name__ == "__main__":
    __main__()