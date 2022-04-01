# How to use the repository

First clone the repository into the desired folder:

```
$ git clone https://github.com/aliciafmachado/sac.git
```

Now, create a conda env with the requirements provided, and then start using the code.

```
$ conda create --name sac-jax --python=3.9
$ conda install pip 
$ conda activate sac-jax
$ pip install -r requirements.txt
$ pip install -e .
```

Other than that, you should also install mujoco. Installation instructions can be found on the `mujoco-py` repository: https://github.com/openai/mujoco-py. 

Then, you can start training and running the sac agent.

## Colab version

You can also run in colab. We leave a notebook example called `Run_SAC.ipynb` that calls training and evaluation on the sac agent for Reacher environment.