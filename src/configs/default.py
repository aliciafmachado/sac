"""
Default hyperparameters for SAC agent.
Runs on Pendulum-v0 environment.

Some of the hyperparameters used here were taken from:
https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml
"""

import ml_collections


def get_config():
    """
    Get the default configuration.
    """
    config = ml_collections.ConfigDict()
    # Batch size
    config.batch_size = 256
    
    # Alpha for rescaling rewards
    config.scale_reward = 1.

    # Learning rate
    config.lr = 1e-3

    # Learning rate for the policy
    config.p_lr = config.lr

    # Learning rate for the value function
    config.v_lr = config.lr

    # Learning rate for the q networks
    config.q_lr = config.lr

    # Choose environment
    # check main.py to see mapping of integers to envs
    config.env_idx = 0

    # Minimum on buffer size before training
    # CAUTION: not tuned
    config.min_buffer_capacity = 5000

    # Use exploratory policy for the same number of steps
    # as the config min buffer
    config.exp_policy_steps = config.min_buffer_capacity

    # Number of updates when doing the update on the nns
    config.number_updates = 1

    # Number steps until updating again
    config.nb_updated_transitions = 1

    # Total number of steps in the environment
    config.num_total_steps = 20000

    # Seed
    config.seed = 42

    # Gamma
    config.gamma = 0.99

    # Replay buffer capacity
    config.replay_buffer_capacity = int(1e6)

    # Hyperparameter for update of target network
    config.tau = 0.005

    # Number of episodes for training if not using nb of steps
    # CAUTION: not tuned
    # config.num_episodes = 1000

    # Eval frequency
    config.eval_frequency = 1000

    # Eval episodes
    config.eval_episodes = 5

    return config