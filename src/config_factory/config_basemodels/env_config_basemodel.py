import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EnvConfigBaseModel(BaseModel):
    """Environment configuration base model."""

    default_max_goal_range: int = Field(description="Default maximum goal range.")
    default_max_steps: int = Field(description="Default maximum steps.")
    default_start_state: int = Field(description="Default start state.")
    valid_actions: list[int] = Field(description="Valid actions.")
    seed: int | None = Field(default=None, description="Random seed.")
