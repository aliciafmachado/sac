"""
Script for training and saving the agent.
"""
import jax
import itertools
import time

# TODO: implement main loop of interaction between agent and environment

# TODO: Discuss buffer update
def train( environment,
                      agent,
                      rng,
                      num_episodes=None,
                      num_steps=None,
                      buffer_capacity=50,
                      number_updates=5,
                      batch_size=10,
                      nb_updated_transitions=2
                      ):
  """Perform the interaction loop.

  Run the environment loop for `num_episodes` episodes. Each episode is itself
  a loop which interacts first with the environment to get an observation and
  then give that observation to the agent in order to retrieve an action. Upon
  termination of an episode a new episode will be started. If the number of
  episodes is not given then this will interact with the environment
  infinitely.

  Args:
    environment: dm_env used to generate trajectories.
    agent: object selecting actions, updating parameters and storing losses.
    rng: random generation seed
    num_episodes: number of episodes to run the loop for. If `None` (default),
    runs without limit.
    num_steps: number of episodes to run the loop for. If `None` (default), runs
    without limit.
    buffer_capacity: minimum number of samples before updating the model
    number_updates: number of updates from the same buffer
    batch_size: size of the training batch
    nb_updated_transitions: (after the buffer is filled completely) number of updated transitions to make before updating the model
  """
  # logger = loggers.TerminalLogger(label=label, time_delta=logger_time_delta)
  iterator = range(num_episodes) if num_episodes else itertools.count()
  all_logs = []
  all_returns = []

  num_total_steps = 0
  # initialiaze agent and LearnerState
  learner_state = agent.initialize()

  for episode in iterator:

    # Reset any counts and start the environment.
    start_time = time.time()
    episode_steps = 0
    episode_return = 0
    episode_loss_q = 0
    episode_loss_pi = 0
    episode_loss_v = 0

    # Reset environment
    timestep = environment.reset()

    # number of updated transitions
    nb_up_transitions = 0

    # Run an episode.
    while not timestep.last():

      # initialiaze obseration
      obs = timestep.observation

      # Generate an action from the agent's policy and step the environment.
      mu, sigma = agent.apply_policy(learner_state.params.policy, obs)
      stand_gaussian = jax.random.normal(rng, (agent.action_dim,))
      action = (mu + sigma @ stand_gaussian)
      timestep = environment.step(action)

      # store transition
      agent.buffer.store(obs,  action, timestep.reward, timestep.observation, False)
      nb_up_transitions += 1

      if agent.buffer.__len__() >= buffer_capacity and nb_up_transitions >= nb_updated_transitions:

          nb_up_transitions = 0
          for nb_updates in range(number_updates):

              transitions = agent.buffer.sample(batch_size)
              learner_state, logs = agent._update_fn(learner_state, transitions)

              all_logs.append(logs)

      # Book-keeping.
      # episode_steps += 1
      # num_total_steps += 1
      episode_return += timestep.reward



    print(f'episode = {episode}/{num_episodes} ... episode return = {episode_return}')
    # store returns
    all_returns.append(episode_return)

  return all_returns, all_logs, num_total_steps, learner_state
