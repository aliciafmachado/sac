"""
Default hyperparameters for SAC agent.
Runs on Pendulum-v0 environment.

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
    config.num_total_steps = 1e5

    # Gamma
    config.gamma = 0.99

    # Replay buffer capacity
    config.replay_buffer_capacity = int(1e6)

    # Hyperparameter for update of target network
    config.tau = 0.005

    return config