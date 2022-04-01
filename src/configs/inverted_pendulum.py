"""
Default hyperparameters for Inverted pendulum env.

The hyperparameters were based on rl-baselines3-zoo:
https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml
and on the original SAC paper to get better and more stable results.
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

    # Config lr
    config.lr = 7.3e-4

    # Learning rate for the policy
    config.p_lr = config.lr

    # Learning rate for the value function
    config.v_lr = config.lr

    # Learning rate for the q networks
    config.q_lr = config.lr

    # Choose environment
    # check main.py to see mapping of integers to envs
    config.env_idx = 1

    # Minimum on buffer size before training
    config.min_buffer_capacity = 5000

    # Use exploratory policy for the same number of steps
    # as the config min buffer
    config.exp_policy_steps = config.min_buffer_capacity

    # Number of updates when doing the update on the nns
    config.number_updates = 8

    # Number steps until updating again
    config.nb_updated_transitions = 8

    # Total number of steps in the environment
    config.num_total_steps = int(1e5)

    # Gamma
    config.gamma = 0.98

    # Replay buffer capacity
    config.replay_buffer_capacity = int(3e4)

    # Hyperparameter for update of target network
    config.tau = 0.005

    # Eval frequency
    config.eval_frequency = 100

    # Eval episodes
    config.eval_episodes = 5

    return config
