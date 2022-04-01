import copy
from src.replay_buffers.buffer import ReplayBuffer
import acme
from src.envs.pendulum import PendulumEnv
import tree
from src.agents.random_agent import RandomAgent
from src.utils.training_utils import Transitions
import jax


def pre_fill(rng, env, buffer, n_trajectories):
  random_agent = RandomAgent(acme.make_environment_spec(env))
  for _ in range(n_trajectories):
    ts = env.reset()
    obs = ts.observation
    while True:
        rng, key = jax.random.split(rng, 2)
        batched_observation = tree.map_structure(lambda x: x[None], ts.observation)
        a = random_agent.get_action(key, batched_observation)[0]
        ts = env.step(a)
        last_obs = copy.deepcopy(obs)
        obs = ts.observation
        action = a
        done = ts.last()
        reward = ts.reward
        buffer.store(state=last_obs, action=action, reward=reward, next_state=obs, done=done)
        if ts.last():
            break

    return buffer


rng = jax.random.PRNGKey(0)
env = PendulumEnv(for_evaluation=False)
environment_spec = acme.make_environment_spec(env)
action_dim = environment_spec.actions.shape[-1]
observation_dim = environment_spec.observations.shape[-1]
buffer = ReplayBuffer(size_=10000, featuredim_=observation_dim, actiondim_=action_dim)
buffer_2 = pre_fill(rng, env, buffer, 1)
print(buffer_2.sample(10).actions)