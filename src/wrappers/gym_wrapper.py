"""Gym wrapper for the environment."""

import gymnasium as gym
import numpy as np
from base_env.env import DiscreteGoalReach1D
from base_env.type_defs import DoneType, RewardType, StateType
from numpy.typing import NDArray

InfoType = dict
ResetReturnType = tuple[NDArray[np.int32], InfoType]
StepReturnType = tuple[NDArray[np.int32], RewardType, DoneType, DoneType, InfoType]


class GymWrapper(gym.Env):
    """Gym wrapper for the environment."""

    def __init__(self, env: DiscreteGoalReach1D):
        self._env = env
        self.observation_space = gym.spaces.Box(
            low=-env.max_goal, high=env.max_goal, shape=(1,), dtype=np.int32
        )
        self.action_space = gym.spaces.Discrete(env.n_actions)

    @property
    def base_env(self) -> DiscreteGoalReach1D:
        """The base environment."""
        return self._env

    def reset(self, **kwargs) -> ResetReturnType:  # pylint: disable=unused-argument
        """Reset the environment."""
        observation = self.base_env.reset()
        info_dict = self._get_info_dict()
        return observation, info_dict

    def step(self, action: int) -> StepReturnType:
        """Take a step in the environment."""
        state, reward, truncated = self.base_env.step(action)
        observation = self._get_observation(state)
        terminated = self._get_terminated()
        info_dict = self._get_info_dict()
        return observation, reward, terminated, truncated, info_dict

    def render(self) -> None:
        """Render the environment."""
        self.base_env.render()

    def _get_observation(self, state: StateType) -> NDArray[np.int32]:
        """Get the observation."""
        return np.array([state], dtype=np.int32)

    def _get_terminated(self) -> bool:
        """Get the terminated flag."""
        return False

    def _get_info_dict(self) -> dict:
        """Get the info dictionary."""
        return {
            "goal": self.base_env.goal,
            "step_count": self.base_env.step_count,
            "max_steps": self.base_env.max_steps,
            "max_goal": self.base_env.max_goal,
        }
