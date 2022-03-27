"""
Unit test for SAC agent.
"""


from src.agents.sac import SAC
from src.envs.inverted_pendulum import InvertedPendulumEnv
import acme
import tree
import ml_collections


def __main__():
    n_simulations = 5
    render = True
    env = InvertedPendulumEnv(for_evaluation=True)
    sac_agent = SAC(acme.make_environment_spec(env), ml_collections.ConfigDict())

    # for _ in range(n_simulations):
    #     ts = env.reset()
    #     while True:
    #         batched_observation = tree.map_structure(lambda x: x[None], ts.observation)
    #         a = random_agent.batched_actor_step(batched_observation)[0]
    #         ts = env.step(a)
    #         if render:
    #             env._env.render()
    #         if ts.last():
    #             break

    # env.close()
    # print("Done.")

if __name__ == "__main__":
    __main__()