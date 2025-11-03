import time
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import gymnasium_robotics
from gymnasium_robotics.envs.fetch.pick_and_place import MujocoFetchPickAndPlaceEnv
from gymnasium_robotics.envs.fetch.reach import MujocoFetchReachEnv
from ray.rllib.algorithms.algorithm import Algorithm
from ray import tune
from wrapper.dict_to_array_obs_wrapper import DictToArrayObsWrapper

fetch_class = MujocoFetchReachEnv

def env_creator(config):
    env = TimeLimit(DictToArrayObsWrapper(fetch_class(reward_type="dense", render_mode="human")), max_episode_steps=50)
    return env


tune.register_env("CustomEnv", env_creator)
checkpoint_dir = "/home/a-akakzia/ray_results/sac-training/SAC_CustomEnv_6c95c_00000_0_2025-09-23_19-04-54/checkpoint_000019"
algo = Algorithm.from_checkpoint(checkpoint_dir)
for _ in range(10):
    results = algo.evaluate()
print(results)
# env: TimeLimit = TimeLimit(DictToArrayObsWrapper(MujocoFetchPickAndPlaceEnv(reward_type="dense")), max_episode_steps=50)
# observation, info = env.reset()
# print(observation)
# step_time: float = 0
# for _ in range(env._max_episode_steps):
#     action = env.action_space.sample()
#     time_start = time.time()
#     observation, reward, terminated, truncated, info = env.step(action)
#     print(reward)
#     step_time += time.time() - time_start
#     if terminated or truncated:
#         observation, info = env.reset()
# env.close()
# print(f"Steps per second: {env._max_episode_steps / step_time}")