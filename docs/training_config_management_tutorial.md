# RLlib Training Configuration Management

This tutorial provides a comprehensive walkthrough of the configuration system used in the `scripts/run_training.py` script. We'll explore how Hydra's composition system and Pydantic's validation work together to create a robust, type-safe training configuration pipeline.

---

## 📚 1. Configuration Architecture Overview

The training script leverages a hierarchical configuration system that separates concerns across multiple configuration files:

- **Main Training Config**: Orchestrates the entire training setup
- **Parameter Space Config**: Defines algorithm-specific hyperparameters 
- **Run Config**: Controls Ray Tune execution parameters
- **Environment Config**: Specifies environment settings

This modular approach enables easy experimentation and ensures configuration consistency across training runs.

> **💡 Info Point - Separation of Concerns**: By splitting configurations into logical groups, we achieve better maintainability, reusability, and clarity. Each config file has a single responsibility and can be modified independently.

---

## 📚 2. Main Training Configuration

The entry point for our configuration system is `configs/training/training_config.yaml`:

```4:15:configs/training/training_config.yaml
defaults: 
  - _self_
  - param_space: sac_config 
  - run: run_config 

_target_: config_factory.config_basemodels.training_config_basemodel.TrainingConfigBaseModel

trainable: SAC
local_mode: false

hydra:
  output_subdir: null
  run:
    dir: .
```

This configuration demonstrates several key concepts:

> **💡 Info Point - Hydra Composition**: The `defaults` section tells Hydra to compose configurations from multiple sources. Here, `param_space: sac_config` means "load `sac_config.yaml` from the `param_space/` subdirectory and assign it to the `param_space` key".

> **💡 Info Point - Pydantic Integration**: The `_target_` field specifies which Pydantic model class should be used to validate and instantiate this configuration. This provides runtime type checking and validation.

The training configuration is validated by `TrainingConfigBaseModel`:

```13:20:src/config_factory/config_basemodels/training_config_basemodel.py
class TrainingConfigBaseModel(BaseModel):
    """Training configuration base model."""

    trainable: Literal["SAC"] = Field(description="Trainable name.")
    local_mode: bool = Field(description="Local mode.")
    param_space: SACConfigBaseModel = Field(description="SAC configuration.")
    run: RunConfigBaseModel = Field(description="Run configuration.")
```

---

## 📚 3. Parameter Space Configuration

The parameter space configuration (`configs/training/param_space/sac_config.yaml`) defines all SAC-specific hyperparameters:

```1:31:configs/training/param_space/sac_config.yaml
defaults:
  - _self_
  - ../../environment/env_config.yaml@environment

_target_: config_factory.config_basemodels.sac_config_basemodel.SACConfigBaseModel


env_runners: 
  num_env_runners: 1
  rollout_fragment_length: auto

training: 
  actor_lr: 0.01
  critic_lr: 0.01
  alpha_lr: 0.01
  initial_alpha: 0.5
  gamma: 0.99 
  tau: 0.005
  train_batch_size_per_learner: 128
  target_network_update_freq: 2 

reporting: 
  min_sample_timesteps_per_iteration: 500

evaluation: 
  evaluation_config: 
    explore: false 
  evaluation_num_env_runners: 1 
  evaluation_duration: 5
  evaluation_interval: 1 
  evaluation_duration_unit: episodes
```

> **💡 Info Point - Package Notation**: The line `../../environment/env_config.yaml@environment` uses Hydra's package notation. It loads the environment config from a relative path and assigns it to the `environment` key in the final configuration.

This configuration is structured into logical sections:
- **env_runners**: Controls parallel environment execution
- **training**: Core SAC algorithm hyperparameters
- **reporting**: Training progress reporting settings
- **evaluation**: Policy evaluation configuration

The corresponding base model provides validation:

```5:12:src/config_factory/config_basemodels/sac_config_basemodel.py
class SACConfigBaseModel(BaseModel):
    """SAC configuration base model."""

    environment: EnvConfigBaseModel = Field(description="Environment configuration.")
    env_runners: dict = Field(description="Environment runners configuration.")
    training: dict = Field(description="Training configuration.")
    reporting: dict = Field(description="Reporting configuration.")
    evaluation: dict = Field(description="Evaluation configuration.")
```

---

## 📚 4. Environment Configuration

The environment configuration (`configs/environment/env_config.yaml`) specifies environment-specific settings:

```1:10:configs/environment/env_config.yaml
# Environment configuration
defaults: 
  - _self_

_target_: config_factory.config_basemodels.env_config_basemodel.EnvConfigBaseModel

name: TruncatedFetchReachEnv
max_episode_steps: 50
render_mode: "human"
reward_type: dense
```

This configuration controls:
- **name**: The environment class to instantiate
- **max_episode_steps**: Episode length limit
- **render_mode**: Visualization settings
- **reward_type**: Reward function variant

> **💡 Info Point - Environment Abstraction**: By separating environment configuration, we can easily swap between different environments or modify environment parameters without touching the core training logic.

---

## 📚 5. Run Configuration

The run configuration (`configs/training/run/run_config.yaml`) manages Ray Tune execution parameters:

```1:15:configs/training/run/run_config.yaml
# Run time configuration 

defaults: 
  - _self_

_target_: config_factory.config_basemodels.run_config_basemodel.RunConfigBaseModel

name: rllib-training

stop: 
  training_iteration: 10 

checkpoint_config:
  checkpoint_frequency: 1 
  checkpoint_at_end: true
```

This configuration defines:
- **name**: Experiment name for tracking
- **stop**: Training termination criteria
- **checkpoint_config**: Model checkpointing behavior

The validation model ensures type safety:

```4:9:src/config_factory/config_basemodels/run_config_basemodel.py
class RunConfigBaseModel(BaseModel):
    """Run time configuration base model."""

    name: str = Field(description="Name of the run.")
    stop: dict = Field(description="Stop criteria.")
    checkpoint_config: dict = Field(description="Checkpoint configuration.")
```

---

## 📚 6. Configuration Loading and Instantiation

The training script loads and processes these configurations through a systematic workflow:

```6:21:scripts/run_training.py
def run_training():
    """Run training."""
    config_path = get_root_path() / "configs" / "training"

    config = load_and_instantiate_hydra_config(config_path=config_path, config_name="training_config")

    assert isinstance(config, TrainingConfigBaseModel)

    tuner = config.to_tuner()

    ray.init(local_mode=config.local_mode)

    tuner.fit()
    
    ray.shutdown()
```

The loading process follows these steps:

1. **Path Resolution**: Determine the configuration directory
2. **Hydra Composition**: Load and merge all referenced configurations
3. **Pydantic Validation**: Instantiate the configuration using the specified base model
4. **Type Assertion**: Verify the configuration type for safety
5. **Conversion**: Transform the configuration into Ray Tune components

> **💡 Info Point - Configuration Flow**: The `load_and_instantiate_hydra_config` function handles both Hydra's composition magic and Pydantic's instantiation. This single function call produces a fully validated, type-safe configuration object.

---

## 📚 7. Configuration Transformation

The `TrainingConfigBaseModel` provides methods to convert the configuration into RLlib components:

```21:40:src/config_factory/config_basemodels/training_config_basemodel.py
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
```

```52:63:src/config_factory/config_basemodels/training_config_basemodel.py
def to_tuner(self) -> Tuner:
    """Convert the training configuration to a ray tune tuner."""

    rllib_config = self.to_rllib_config()

    run_config = self.to_run_config()

    return Tuner(
        trainable=self.trainable,
        param_space=rllib_config.to_dict(), 
        run_config=run_config,
    )
```

These transformation methods bridge the gap between our configuration system and RLlib's training infrastructure.

> **💡 Info Point - Configuration Transformation**: The base model acts as an adapter, converting our structured configuration format into the specific API calls required by RLlib. This separation allows us to maintain clean, readable configurations while still leveraging RLlib's powerful training capabilities.

---

## 📚 8. Benefits of This Configuration System

This configuration architecture provides several key advantages:

### **Modularity**
Each configuration file has a single responsibility, making it easy to modify specific aspects without affecting others.

### **Type Safety** 
Pydantic validation catches configuration errors early, preventing runtime failures due to invalid parameters.

### **Composability**
Hydra's composition system allows mixing and matching different configuration components for different experiments.

### **Reproducibility**
All training parameters are explicitly defined and version-controlled, ensuring reproducible experiments.

### **Maintainability**
The structured approach makes it easy to understand, modify, and extend the configuration system as requirements evolve.

> **💡 Info Point - Production-Ready Configuration**: This system strikes a balance between flexibility and robustness. It's sophisticated enough for complex research workflows while remaining accessible for everyday experimentation.

---

## 📚 9. Configuration Workflow Summary

The complete configuration workflow in the training script follows this pattern:

1. **Composition**: Hydra assembles configurations from multiple files
2. **Validation**: Pydantic validates the composed configuration
3. **Instantiation**: The configuration is converted to a type-safe Python object
4. **Transformation**: The configuration is converted to RLlib components
5. **Execution**: Ray Tune runs the training with the specified parameters

This systematic approach ensures that configuration errors are caught early and that all training runs are properly parameterized and reproducible.

## 📚 10. Your turn

**📢 Action Points**:  Take ownership of the different pydantic basemodels used in this part, add some model validators.
