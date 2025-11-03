import logging
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RolloutConfigBaseModel(BaseModel):
    """Rollout configuration base model.

    Args:
        rollout_unit: Can either be episodes, steps or time (in seconds)
        rollout_unit_value: Number of rollout units to perform.
    """

    rollout_unit: Literal["episodes", "steps", "time"] = Field(
        default="episodes",
        description="Can either be episodes, steps or time (in seconds)",
    )
    rollout_unit_value: int = Field(
        default=1, ge=0, description="Number of rollout units to perform."
    )
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level, must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )

    @field_validator("logging_level", mode="before")
    @classmethod
    def set_logging_level(cls, logging_level: str) -> str:
        """Validate the logging level."""
        # log_level = getattr(logging, self.logging_level)
        logging.basicConfig(
            level=logging_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return logging_level
