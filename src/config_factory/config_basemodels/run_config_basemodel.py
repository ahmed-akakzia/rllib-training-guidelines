from pydantic import BaseModel, Field


class RunConfigBaseModel(BaseModel):
    """Run time configuration base model."""

    name: str = Field(description="Name of the run.")
    stop: dict = Field(description="Stop criteria.")
    checkpoint_config: dict = Field(description="Checkpoint configuration.")