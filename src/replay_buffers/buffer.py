import chex
import jax.numpy as jnp
import numpy as np
from src.utils.training_utils import Transitions



class ReplayBuffer:

    def __init__(self, size_: int, featuredim_: int, actiondim_: int):

        self.__size = size_
        self.__counter = 0
        self.__feature_dim = featuredim_
        self.__action_dim = actiondim_
        self.__states = jnp.zeros((size_, featuredim_))
        self.__next_states = jnp.zeros((size_, featuredim_))
        self.__actions = jnp.zeros((size_, actiondim_))
        self.__rewards = jnp.zeros(size_)
        self.__dones = jnp.zeros(size_, dtype=bool)
        self.__output = Transitions()

    def store(self, state: chex.Array, action: chex.Array, reward: float, next_state: chex.Array, done: bool):

        # Check if buffer is full
        if self.__counter >= self.__size:
            index = self.__counter % self.__size

        else:
            index = self.__counter

        self.__states.at[index].set(state)
        self.__next_states.at[index].set(next_state)
        self.__actions.at[index].set(action)
        self.__rewards.at[index].set(reward)
        self.__dones.at[index].set(done)

        # Increment counter
        self.__counter += 1

        return 0

    def sample(self, batch_size: int) -> Transitions:

        # sample random indexes
        memory_ = min(self.__size, self.__counter)
        idx_ = np.random.choice(memory_, batch_size)

        # return batch sample
        self.__output.observations = self.__states[idx_]
        self.__output.actions = self.__actions[idx_]
        self.__output.rewards = self.__rewards[idx_]
        self.__output.next_observations = self.__next_states[idx_]
        self.__output.dones = self.__dones[idx_]

        return self.__output

    def buffer_state(self):

        if self.__counter < self.__size:
            print('Buffer not full')

        else:
            print('Buffer full')















