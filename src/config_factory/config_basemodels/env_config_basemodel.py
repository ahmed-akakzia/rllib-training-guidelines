from typing import Any, Callable, Literal

from gymnasium.wrappers import TimeLimit
from gymnasium_robotics.envs.fetch.reach import MujocoFetchReachEnv
from pydantic import BaseModel, Field
from ray import tune


class EnvConfigBaseModel(BaseModel):
    """Environment configuration base model."""

    name: str = Field(description="Name of the environment.")
    max_episode_steps: int = Field(description="Maximum number of steps per episode.")
    render_mode: Literal["human", "rgb_array"] | None = Field(description="Render mode.")
    reward_type: Literal["sparse", "dense"] = Field(description="Reward type.")


    @staticmethod
    def env_creator(config: Any) -> Callable:
        """Create an environment."""
        env_config = EnvConfigBaseModel.model_validate(config)
        env = MujocoFetchReachEnv(reward_type=env_config.reward_type)
        return TimeLimit(env, max_episode_steps=env_config.max_episode_steps)
    

    def model_post_init(self, __context: Any) -> None:
        """Register the environment creator within the post init hook."""
        tune.register_env(self.name, self.env_creator)