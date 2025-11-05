# RLlib Training Guidelines

The objective of this training is to gain knowledge of [RLlib](https://docs.ray.io/en/latest/rllib/index.html), a widely used open source library for reinforcement learning (RL). RLlib is used in many projects at InstaDeep thanks to its production-level support, maintainability and relatively easy usage.

Throughout this training, your task is to solve the [Fetch Reach](https://robotics.farama.org/envs/fetch/reach/) problem with the Soft Actor-Critic ([SAC](https://arxiv.org/abs/1801.01290)) algorithm with continuous actions. This is a simply goal-conditioned problem where you need to train a single policy to reach several different goals. At the end of the training, you should be able to train an agent that does the following: 

<p align="center">
  <img src="images/robot.gif" alt="Funny Meme" width="500">
</p>

To accomplish this mission, you will need to **customize many components in RLlib, augment your solution with connectors, launch and evaluate experiments**. We have broken this exercise into several steps, which can be found in the issues directory. The main steps are outlined below: 

+ [Setup dev environment and install dependencies.](docs/setup_env.md) 
+ [Configuring Rollouts with Hydra and Pydantic.](docs/config_management_tutorial.md)
+ [RLlib Training Configuration Management.](docs/training_config_management_tutorial.md) 
+ [Understanding RLlib Catalogs: Structured Observations in SAC.](docs/rllib_catalogs.md) 
+ [Understanding RLlib Connectors: Data Transformation Pipelines.](docs/rllib_connectors.md)
+ [Adding the Neptune Logger Callback to RLlib.](docs/neptune_logging.md) 
+ [Training with SAC to Reach Multiple Goals.](docs/training_and_tuning.md)
