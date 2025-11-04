from typing import Literal

from pydantic import BaseModel, Field


class EnvConfigBaseModel(BaseModel):
    """Environment configuration base model."""

    max_episode_steps: int = Field(description="Default maximum goal range.")
    render_mode: Literal["human", "rgb_array"] | None
    reward_type: Literal["sparse", "dense"]
