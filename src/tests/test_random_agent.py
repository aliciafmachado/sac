"""
Unit test for random agent.
"""

from src.agents.random_agent import RandomAgent
from src.envs.inverted_pendulum import InvertedPendulumEnv
from src.envs.reacher import ReacherEnv
import acme
import tree
import pickle
import jax


def __main__():
    rng = jax.random.PRNGKey(0)
    n_simulations = 10
    render = False
    env = ReacherEnv(for_evaluation=False)
    random_agent = RandomAgent(acme.make_environment_spec(env))
    rewards = []

    for _ in range(n_simulations):
        ts = env.reset()
        reward = 0
        while True:
            rng, key = jax.random.split(rng, 2)
            batched_observation = tree.map_structure(lambda x: x[None], ts.observation)
            a = random_agent.get_action(key, batched_observation)[0]
            ts = env.step(a)
            if render:
                env._env.render()
            reward += ts.reward
            if ts.last():
                break
        rewards.append(reward)

    env.close()
    with open("results/random.pickle", "wb") as f:
      pickle.dump(rewards, f)
    print(rewards)
    print("Done.")

if __name__ == "__main__":
    __main__()