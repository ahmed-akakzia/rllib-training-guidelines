import logging
import random

from config_factory.config_basemodels.env_config_basemodel import EnvConfigBaseModel
from config_factory.config_basemodels.rollout_config_basemodel import (
    RolloutConfigBaseModel,
)
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class ConfigBaseModel(BaseModel):
    """Base configuration model."""

    environment: EnvConfigBaseModel
    rollout: RolloutConfigBaseModel
    seed: int | None = Field(description="Random seed.")

    @field_validator("seed", mode="before")
    @classmethod
    def set_seed(cls, seed: int | None) -> int:
        """Validate the random seed."""
        if seed is None:
            logger.info("No seed provided, generating a random seed.")
            return random.randint(0, 1000000)
        return seed

    @model_validator(mode="after")
    def set_environment_seed(self) -> "ConfigBaseModel":
        """Set the seed for the environment."""
        self.environment.seed = self.seed
        return self
