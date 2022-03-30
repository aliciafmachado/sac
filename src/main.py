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
import pickle
import os
import shutil


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    'config',
    "src/configs/default.py",
    'File path to the default configuration file.',
    lock_config=True)
flags.DEFINE_string('save_pth', 'results', 'Path to folder where to save the model')
flags.DEFINE_string('experiment', 'experiment_0', 'Name of the experiment')


environments = {
  0: PendulumEnv,
  1: InvertedPendulumEnv,
  2: ReacherEnv,
}


def main(argv):
    print(f'Model will be saved in {FLAGS.save_pth}/{FLAGS.experiment}')

    # Create folder for saving model
    full_path = os.path.join(FLAGS.save_pth, FLAGS.experiment)
    
    if not os.path.exists(FLAGS.save_pth):
      os.mkdir(FLAGS.save_pth)

    if os.path.exists(full_path):
      answer = input("Saving the model will overwrite folder named {}. Continue (y/n)?".format(full_path))
      if answer.lower() in ["y", "yes"]:
        shutil.rmtree(full_path)
      else:
        print("Change experiment name please and repeat.")
        return
    
    os.mkdir(full_path)

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
                      num_episodes=config.num_episodes,
                      num_steps=None, # it was 1 before
                      min_buffer_capacity=config.min_buffer_capacity,
                      number_updates=config.number_updates,
                      batch_size=config.batch_size,
                      nb_updated_transitions=config.nb_updated_transitions,
                      exploratory_policy_steps=config.exp_policy_steps,
                      )

    metrics = {
      'all_returns': all_returns,
      'all_logs': all_logs,
      'num_total_steps': num_total_steps,
    }

    model = {
      'config': config,
      'learner_state': learner_state,
    }

    mm = {
      'metrics': metrics,
      'model': model,
    }

    # Save model and metrics
    with open(os.path.join(full_path, FLAGS.experiment + "_mm.pickle"), 'wb') as f:
      pickle.dump(mm, f)

    print('done')

if __name__ == '__main__':
  app.run(main)