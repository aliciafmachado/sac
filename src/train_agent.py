"""
Script for training and saving the agent.
"""
import jax
import numpy as np


def train(environment, eval_environment, 
                      agent,
                      rng,
                      min_buffer_capacity=50,
                      number_updates=5,
                      batch_size=10,
                      nb_updated_transitions=2,
                      exploratory_policy_steps=200,
                      nb_training_steps=None,
                      eval_frequency=10000,
                      eval_episodes=5,
                      verbose=True,
                      verbose_frequency=100,
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
    eval_environment: dm_env used for evaluation of the agent
    agent: object selecting actions, updating parameters and storing losses.
    rng: random generation seed
    num_steps: number of episodes to run the loop for. If `None` (default), runs
    without limit.
    min_buffer_capacity: minimum number of samples before updating the model
    number_updates: number of updates from the same buffer
    batch_size: size of the training batch
    nb_updated_transitions: (after the buffer is filled completely) number of updated transitions to make before updating the model
    exploratory_policy_steps: nb of steps to explore using uniformly sampled actions
    nb_training_steps: nb of steps to interact with the env
    eval_frequency: frequency of evaluation
    eval_episodes: number of episodes to evaluate the agent
    verbose: set true if you want to debug
    verbose_frequency: frequency of verbose print
  """
  # Metrics logging
  all_logs = []
  eval_rewards = []

  num_total_steps = 0

  # initialiaze agent and LearnerState
  learner_state = agent.initialize()

  # number of updated transitions
  nb_up_transitions = 0

  # Initialize environment
  timestep = environment.reset()

  while(num_total_steps < nb_training_steps):

    obs = timestep.observation
         
    # Sample action from initial exploratory policy
    if num_total_steps < exploratory_policy_steps:
        action = environment._env.action_space.sample()

    else:
        # Generate an action from the agent's policy and step the environment.
        rng, key = jax.random.split(rng, 2)
        action = agent.get_action(key, learner_state.params.policy, obs)
        
    timestep = environment.step(action)
    
    # Increaser number of total steps
    num_total_steps += 1

    # store transition
    agent.buffer.store(obs,  action, timestep.reward, timestep.observation, timestep.last())
    nb_up_transitions += 1

    if agent.buffer.__len__() >= min_buffer_capacity and nb_up_transitions >= nb_updated_transitions:
        nb_up_transitions = 0
        for _ in range(number_updates):

            transitions = agent.buffer.sample(batch_size)
            learner_state, logs = agent.update_fn(learner_state, transitions)

            all_logs.append(logs)

    if timestep.last():
        timestep = environment.reset()

    # Log if debugging
    if verbose and num_total_steps % verbose_frequency == 0:
      if agent.buffer.__len__() >= min_buffer_capacity:
        mean_loss_q = np.mean([a['loss_q'] for a in all_logs[-verbose_frequency:]])
        mean_loss_pi = np.mean([a['loss_pi'] for a in all_logs[-verbose_frequency:]])
        mean_loss_v = np.mean([a['loss_v'] for a in all_logs[-verbose_frequency:]])
        entropy = np.mean([a['entropy'] for a in all_logs[-verbose_frequency:]])
        print(f'loss q:{mean_loss_q}\nloss pi:{mean_loss_pi}\nloss_v:{mean_loss_v}\nentropy:{entropy}')
      else:
        print("Filling buffer and exploring...")
      print(f'nb of steps:{num_total_steps}')

    # Do evaluation
    if num_total_steps % eval_frequency == 0:
        ev_rewards = []
        for _ in range(eval_episodes):
            timestep = eval_environment.reset()
            rewards_episode = 0

            while not timestep.last():
                # We don't need to split the key since it is deterministic
                key = None
                timestep = eval_environment.step(agent.get_action(key, 
                        learner_state.params.policy, 
                        timestep.observation, 
                        deterministic=True))
                rewards_episode += timestep.reward

            ev_rewards.append(rewards_episode)
        
        mean_rewards = np.mean(ev_rewards)
        eval_rewards.append(mean_rewards)

        print(f'Evaluation after {num_total_steps} steps: {mean_rewards}')
        print('All rewards: ', ev_rewards)
    
  return eval_rewards, all_logs, num_total_steps, learner_state
