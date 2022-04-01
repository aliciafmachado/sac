"""
Uniform Replay Buffer class.
"""

import chex
import jax.numpy as jnp
import numpy as np
from src.utils.training_utils import Transitions


class ReplayBuffer:
    """
    Uniform Replay Buffer to store transitions and sample them.
    """

    def __init__(self, size_: int, featuredim_: int, actiondim_: int):
        """
        Initialize buffer.

        Args:
            size_: Size of the buffer.
            featuredim_: Dimension of the state.
            actiondim_: Dimension of the action.
        """

        self.__size = size_
        self.__counter = 0
        self.__states = jnp.zeros((size_, featuredim_))
        self.__next_states = jnp.zeros((size_, featuredim_))
        self.__actions = jnp.zeros((size_, actiondim_))
        self.__rewards = jnp.zeros(size_)
        self.__dones = jnp.zeros(size_, dtype=bool)

    def store(self, state: chex.Array, action: chex.Array, 
                    reward: float, next_state: chex.Array, done: bool):
        """
        Store a transition in the replay buffer.
        """

        # Check if buffer is full
        if self.__counter >= self.__size:
            index = self.__counter % self.__size

        else:
            index = self.__counter

        self.__states = self.__states.at[index].set(state)
        self.__next_states = self.__next_states.at[index].set(next_state)
        self.__actions = self.__actions.at[index].set(action)
        self.__rewards = self.__rewards.at[index].set(reward)
        self.__dones = self.__dones.at[index].set(done)

        # Increment counter
        self.__counter += 1

    def sample(self, batch_size: int) -> Transitions:
        """
        Sample a batch of transitions from the replay buffer.
        """

        # sample random indexes
        memory_ = min(self.__size, self.__counter)
        idx_ = np.random.choice(memory_, batch_size)

        # return batch sample
        return Transitions(
          observations = self.__states[idx_],
          actions = self.__actions[idx_],
          rewards = self.__rewards[idx_],
          next_observations = self.__next_states[idx_],
          dones = self.__dones[idx_],
        )

    def buffer_state(self):
        """
        Print buffer state.
        """

        if self.__counter < self.__size:
            print('Buffer not full')

        else:
            print('Buffer full')

    def __len__(self):
        return np.min(self.__counter, self.__size)

