"""
Function for evaluating the agent.
"""
import numpy as np
import jax
import tree


def evaluate(environment,
             agent,
             agent_type,
             nb_episodes,
             learner_state,
             rng):
    """
    Evaluate the agent
    """
    eval_rewards = []

    for i in range(nb_episodes):
        timestep = environment.reset()
        eval_episode = 0

        while not timestep.last():

            obs = timestep.observation
            if agent_type == 'random':
                rng, key = jax.random.split(rng, 2)
                batched_observation = tree.map_structure(lambda x: x[None], obs)
                action = agent.get_action(key, batched_observation)[0]
      
            else:
                # Evaluate deterministically with the agent's policy
                key = None
                action = agent.get_action(key, learner_state.params.policy, 
                                          obs, deterministic=True)                  

            timestep = environment.step(action)
            eval_episode += timestep.reward

        eval_rewards.append(eval_episode)
        print(f'Reward on simulation {i+1}: {eval_episode:.2f}')
    
    return np.array(eval_rewards)
    