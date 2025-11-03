import time
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import gymnasium_robotics
from gymnasium_robotics.envs.fetch.reach import MujocoFetchReachEnv
from wrapper.dict_to_array_obs_wrapper import DictToArrayObsWrapper

# env = gym.make("FetchReach-v4", reward_type="dense")
env: TimeLimit = TimeLimit(DictToArrayObsWrapper(MujocoFetchReachEnv(reward_type="sparse")), max_episode_steps=50)
observation, info = env.reset()
step_time: float = 0
done = False
while not done:
    action = env.action_space.sample()
    time_start = time.time()
    observation, reward, terminated, truncated, info = env.step(action)
    step_time += time.time() - time_start
    done = terminated or truncated
    print(reward)
env.close()
# print(f"Steps per second: {n_steps / step_time}")