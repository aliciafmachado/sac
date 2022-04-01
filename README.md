# sac

Soft Actor-Critic implementation in JAX.

## Installation instructions:

First, create your environment with the requirements on requirements.txt.

Then, you can run an agent using the script `eval.py`, and train an agent with `main.py`.

## Training the agent

In order to train the sac agent, you should execute:

```
$ python main.py --seed [seed] --experiment [name of experiment] --config [config file]
```

In this case, you could use seed equal to 42, experiment "test", and one of the config files in the folder `src/configs`. You can also set an extra flag to where to save the experiments. By default, it creates a folder `results` and saves them there. The model and metrics are saved in the same file in a pickle fashion.

## Running the agent

If you want to evaluate outside the training loop (with sac or a random agent), you can use `eval.py`:

```
$ eval_py --agent [agent] --agent_path [agent_path] --env [env_idx]
```

Notice that the indexes of the environments can be found in the `src/utils/training_utils.py` file.

We also trained a PPO agent to compare our results with, but this wasn't integrated with the rest of the code, and it's done on the notebook `Run_PPO.ipynb` in the `notebooks` folder.