# SAC: Soft Actor-Critic

Soft Actor-Critic implementation in JAX, based on **Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor** from Haarnoja et al.

We use the original algorithm with value, Q and policy networks.

## Installation instructions:

First, create your environment with the requirements on requirements.txt.

Then, you can run an agent using the script `eval.py`, and train an agent with `main.py`.

## Training the agent

In order to train the sac agent, you should execute:

```
$ python main.py --seed [seed] --experiment [name_of_experiment] --config [config_file]
```

In this case, you could use seed equal to 42, experiment "test", and one of the config files in the folder `src/configs`. You can also set an extra flag to where to save the experiments. By default, it creates a folder `results` and saves them there. The model and metrics are saved in the same file in a pickle fashion.

## Running the agent

If you want to evaluate outside the training loop (with sac or a random agent), you can use `eval.py`:

```
$ eval_py --agent [agent] --mm_path [path_to_the_file_with_model] --nb_episodes [nb_episodes] --seed [seed] --env [env_idx]
```

First, the `agent` parameter here should be `random` and `sac`, which are the agent that work with `eval.py`. For `env_idx`, the indexes of the environments can be found in the `src/utils/training_utils.py` file, and notice that you only have to give them as input if you are dealing with a random agent. Otherwise, the eval function simply use the environment saved on the config which is saved with the model on the `mm.pickle` file. When running with `random` agent, you don't need to set the path to a saved file.

The seed that is fed here is use on the environment and from action selection when using a random agent.

We also trained a PPO agent to compare our results with, but this wasn't integrated with the rest of the code, and it's done on the notebook `Run_PPO.ipynb` in the `notebooks` folder.