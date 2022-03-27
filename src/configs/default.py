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
    config.batch_size = 32
    
    # Alpha for rescaling rewards
    config.alpha = 0.2

    # Learning rate for the policy
    config.policy_lr = 3e-4

    # Learning rate for the value function
    config.value_lr = 3e-4

    # Learning rate for the q networks
    config.q_lr = 3e-4

    # TODO: complete and check the hyperparameters

    return config
