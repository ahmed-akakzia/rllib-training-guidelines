from config_factory.config_basemodels.env_config_basemodel import EnvConfigBaseModel
from pydantic import BaseModel, Field


class SACConfigBaseModel(BaseModel):
    """SAC configuration base model."""

    environment: EnvConfigBaseModel = Field(description="Environment configuration.")
    env_runners: dict = Field(description="Environment runners configuration.")
    training: dict = Field(description="Training configuration.")
    reporting: dict = Field(description="Reporting configuration.")
    evaluation: dict = Field(description="Evaluation configuration.")