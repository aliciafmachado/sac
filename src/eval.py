"""
Script for running and visualizing agents.
"""

from absl import flags, app
from ml_collections import config_flags
from src.utils.training_utils import environments, env_names
import jax
import pickle
from src.eval_agent import evaluate
import tensorflow as tf
import acme
from src.agents.sac import SAC
from src.agents.random_agent import RandomAgent


FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 42, 'Seed for simulation')
flags.DEFINE_integer('nb_episodes', 10, 'Number of episodes to run')
flags.DEFINE_integer('agent', 'sac', 'Agent to run: can be `sac` or `random`')
flags.DEFINE_string('mm_pth', 'results/test/test_mm.pickle', 'Path to where the file with the model is saved.')


def main():
    # Open pickle
    with open(FLAGS.mm_pth, "rb") as f:
        mm = pickle.load(f)
    
    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], 'GPU')

    # Get learner state
    ls = mm['model']['learner_state']

    # Get configurations
    config = mm['model']['config']
    env_idx = config.env_idx

    # We create the environment and seed it
    env = environments[int(env_idx)]()
    env._env.seed(seed=FLAGS.seed)
    environment_spec = acme.make_environment_spec(env)
    rng = jax.random.PRNGKey(FLAGS.seed)

    print(f'Evaluating {flags.agent} agent on env {env_names[int(env_idx)]}')

    if flags.agent == 'random':
        agent = RandomAgent(environment_spec)
    elif flags.agent == 'sac':
        rng, key = jax.random.split(rng, 2)
        agent = SAC(key, environment_spec, config)
        _ = agent.initialize()
    else:
        raise ValueError(f'Unknown agent {flags.agent}')

    # Call evaluation
    eval_rewards = evaluate(environment=env,
                            agent=agent,
                            agent_type=flags.agent,
                            nb_episodes=FLAGS.nb_episodes,
                            learner_state=ls,
                            rng=rng)

    print(f'Evaluation done. Average reward: {eval_rewards.mean():.2f}')


if __name__ == '__main__':
  app.run(main)