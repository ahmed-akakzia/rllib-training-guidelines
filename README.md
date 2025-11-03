# RLlib Training Guidelines

The objective of this training is to gain knowledge of [RLlib](https://docs.ray.io/en/latest/rllib/index.html), a widely used open source library for reinforcement learning (RL). RLlib is used in many projects at InstaDeep thanks to its production-level support, maintainability and relatively easy usage.

Throughout this training, your task is to solve the [Fetch Reach](https://robotics.farama.org/envs/fetch/reach/) problem with the Soft Actor-Critic ([SAC](https://arxiv.org/abs/1801.01290)) algorithm with [discrete actions](https://arxiv.org/abs/1910.07207). This is a simply goal-conditioned problem where you need to train a single policy to reach several different goals. At the end of the training, you should be able to train an agent that does the following: 

<p align="center">
  <img src="images/robot.gif" alt="Funny Meme" width="500">
</p>

To accomplish this mission, you will need to **customize many components in RLlib, augment your solution with connectors, launch and evaluate experiments**. We have broken this exercise into several steps, which can be found in the issues directory. The main steps are outlined below: 

+ [Setup dev environment and install dependencies.](docs/setup_env.md) 
+ [Implement a Simple Rollout script with a Random Policy.](docs/rollout_random_agent.md)
+ [Implement DictToArrayObservationWrapper.](docs/observation_wrapper.md) 
+ [Implement an RLlib training script using ray tuner.](docs/training_with_tuner.md) 
+ [Implement a neptune logging callback.](docs/neptune.md)
+ [Leverage EnvToModulePipeline to normalize observations.](docs/env_to_module_pipeline.md) 
+ [Tune and train an agent with dense rewards.](docs/tuning_and_training_agent_dense_reward.md)
+ [Implement a script to load a checkpoint and run offline evaluation.](docs/load_checkpoint.md) 
+ [Implement HER through the LearnerPipeline.](docs/her_with_learner_pipeline.md) 
+ [Implement an evaluation script that compare two checkpoints.](docs/comparing_two_checkpoints.md) 

### Contributing

- [x] Create repo.
- [x] Setup uv.
- [x] Setup environment.
- [ ] Setup RLlib.
- [ ] Setup Training Pipeline.

### Notes: 

no gymnasium robotics in pyproject yaml. 

Setup gymnasium robotics: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/a-akakzia/.mujoco/mujoco210/bin

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia