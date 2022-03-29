"""
Default hyperparameters for SAC agent.
"""

import ml_collections

# TODO: add different configurations depending on the environment
def get_config():
    """
    Get the default configuration.
    """
    config = ml_collections.ConfigDict()
    # Batch size
    config.batch_size = 64
    
    # Alpha for rescaling rewards
    config.scale_reward = 1.

    # Learning rate for the policy
    config.p_lr = 3e-3

    # Learning rate for the value function
    config.v_lr = 3e-3

    # Learning rate for the q networks
    config.q_lr = 3e-3

    # Choose environment
    # check main.py to see mapping of integers to envs
    config.env_idx = 0

    # Seed
    config.seed = 42

    # Gamma
    config.gamma = 0.99

    # Replay buffer capacity
    config.replay_buffer_capacity = int(1e6)

    # Hyperparameter for update of target network
    config.tau = 0.005

    return config
