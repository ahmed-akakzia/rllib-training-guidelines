import numpy as np
import gymnasium as gym
from gymnasium_robotics.envs.fetch.pick_and_place import MujocoFetchPickAndPlaceEnv
from numpy.typing import NDArray



class DictToArrayObsWrapper(gym.Wrapper):
    def __init__(self, env: MujocoFetchPickAndPlaceEnv):
        super().__init__(env)
        n_flattened_dim = sum(value.shape[0] for value in env.observation_space.values())
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n_flattened_dim,))
    
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        transformed_observation = self._transform_obs(observation)
        return transformed_observation, info
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        transformed_observation = self._transform_obs(observation)
        return transformed_observation, reward, terminated, truncated, info
    
    def _transform_obs(self, observation_dict: dict[str, NDArray]) -> NDArray:
        transformed_obs = np.concatenate([value for value in observation_dict.values()])
        return transformed_obs


    