from typing import Literal
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from config_factory.config_basemodels.run_config_basemodel import RunConfigBaseModel
from config_factory.config_basemodels.sac_config_basemodel import SACConfigBaseModel
from pydantic import BaseModel, Field
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.tune import RunConfig, Tuner, CheckpointConfig
from training.catalogs import StructuredObservationSACCatalog
from training.connectors import make_custom_env_to_module_connector_pipeline


class TrainingConfigBaseModel(BaseModel):
    """Training configuration base model."""

    trainable: Literal["SAC"] = Field(description="Trainable name.")
    local_mode: bool = Field(description="Local mode.")
    param_space: SACConfigBaseModel = Field(description="SAC configuration.")
    run: RunConfigBaseModel = Field(description="Run configuration.")

    def to_rllib_config(self) -> AlgorithmConfig:
        """Convert the training configuration to an RLlib algorithm configuration."""
        if self.trainable == "SAC":
            sac_config = SACConfig()
            
            sac_config.environment(self.param_space.environment.name, env_config=self.param_space.environment.model_dump())

            sac_config.env_runners(**self.param_space.env_runners, env_to_module_connector=make_custom_env_to_module_connector_pipeline)

            sac_config.rl_module(rl_module_spec=RLModuleSpec(catalog_class=StructuredObservationSACCatalog))

            sac_config.training(**self.param_space.training)

            sac_config.reporting(**self.param_space.reporting)

            sac_config.evaluation(**self.param_space.evaluation)

            return sac_config
        else:
            raise ValueError(f"Trainable {self.trainable} not supported.")
    
    def to_run_config(self) -> RunConfig:
        """Convert the training configuration to a ray tune run configuration."""
        return RunConfig(
            name=self.run.name,
            stop=self.run.stop,
            checkpoint_config=CheckpointConfig(
                **self.run.checkpoint_config
            ),
        )
    
    def to_tuner(self) -> Tuner:
        """Convert the training configuration to a ray tune tuner."""

        rllib_config = self.to_rllib_config()

        run_config = self.to_run_config()

        return Tuner(
            trainable=self.trainable,
            param_space=rllib_config.to_dict(), 
            run_config=run_config,
        )