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
    # TODO: complete and check the hyperparameters

    return config
