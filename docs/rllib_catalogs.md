# Understanding RLlib Catalogs: Structured Observations in SAC

This tutorial explores RLlib's catalog system through the lens of the `StructuredObservationSACCatalog` implementation. We'll dive deep into how catalogs work, why they're essential for custom observation spaces, and how to implement specialized neural network architectures for reinforcement learning algorithms.

---

## 📚 1. What are RLlib Catalogs?

RLlib catalogs are factory classes that define how to build neural network components for reinforcement learning algorithms. They serve as blueprints that specify:

- **Model architectures** for policy and value networks
- **Encoder configurations** for processing observations
- **Action distributions** for sampling actions
- **Preprocessing logic** for complex observation spaces

> **💡 Info Point - Catalog Purpose**: Catalogs decouple algorithm logic from neural network architecture decisions. This separation allows you to experiment with different model architectures without modifying the core algorithm implementation.

The catalog system enables modular design where you can swap different network architectures by simply changing the catalog class, making experimentation and customization significantly easier.

Please check the code for the core [Catalog](https://github.com/ray-project/ray/blob/master/rllib/core/models/catalog.py) and [SACCatalog](https://github.com/ray-project/ray/blob/master/rllib/algorithms/sac/sac_catalog.py) classes. 

---

## 📚 2. The Challenge: Dictionary Observation Spaces

Many real-world environments produce structured observations in dictionary format, where different sensors or data sources are grouped by keys:

```python
observation = {
    "position": np.array([x, y, z]),
    "velocity": np.array([vx, vy, vz]), 
    "goal": np.array([gx, gy, gz]),
    "gripper": np.array([gripper_state])
}
```

Standard RLlib catalogs expect simple observation spaces (like `Box` spaces with a single tensor). Dictionary spaces require custom handling to:

1. **Validate space structure** - Ensure all sub-spaces are compatible
2. **Aggregate dimensions** - Combine multiple spaces into a unified input
3. **Handle concatenation** - Properly merge observation and action data for Q-functions

> **💡 Info Point - Structured Observations**: Dictionary observation spaces are common in robotics and multi-modal environments where different sensors provide heterogeneous data. Proper handling ensures the neural network receives well-formatted input.

---

## 📚 3. StructuredObservationSACCatalog Overview

Our custom catalog extends the standard `SACCatalog` to handle dictionary observation spaces:

```9:9:src/training/catalogs/structured_observation_sac_catalog.py
class StructuredObservationSACCatalog(SACCatalog):
```

This inheritance means we keep all standard SAC functionality while overriding specific methods to handle structured observations.

The catalog provides two main customizations:
- **Policy encoder configuration** via `_get_encoder_config`
- **Q-function encoder building** via `build_qf_encoder`

> **💡 Info Point - Inheritance Strategy**: By inheriting from `SACCatalog`, we leverage existing SAC-specific logic while customizing only the observation processing parts. This approach minimizes code duplication and maintains compatibility.

---

## 📚 4. Observation Space Validation and Processing

The catalog first validates that dictionary observation spaces meet specific requirements:

```17:24:src/training/catalogs/structured_observation_sac_catalog.py
if (
    isinstance(observation_space, gym.spaces.Dict)
):
    # Chech that all spaces inside the Dict are of Box types and have 1 shape
    for space in observation_space.spaces.values():
        assert isinstance(space, gym.spaces.Box)
        assert len(space.shape) == 1
    aggregated_shape = sum([space.shape[-1] for space in observation_space.spaces.values()])
```

This validation ensures:
- All sub-spaces are `Box` types (continuous values)
- All sub-spaces are 1-dimensional vectors
- We can safely aggregate dimensions by summing shapes

> **💡 Info Point - Dimension Aggregation**: The catalog flattens dictionary observations by concatenating all 1D vectors into a single input tensor. This approach works well when all observation components are of similar scale and importance.

If the observation space doesn't match this pattern, the catalog falls back to the parent class behavior:

```59:60:src/training/catalogs/structured_observation_sac_catalog.py
else:
    return super()._get_encoder_config(observation_space, model_config_dict, action_space)
```

---

## 📚 5. Policy Encoder Configuration

The policy encoder processes observations to create feature representations for action selection. The catalog builds this encoder using `MLPEncoderConfig`:

```25:43:src/training/catalogs/structured_observation_sac_catalog.py
hidden_layer_dims = model_config_dict["fcnet_hiddens"][:-1]
encoder_latent_dim = model_config_dict["fcnet_hiddens"][-1]
encoder_config: ModelConfig = MLPEncoderConfig(
    input_dims=(aggregated_shape,), # TODO: Refine
    hidden_layer_dims=hidden_layer_dims,
    hidden_layer_activation=model_config_dict["fcnet_activation"],
    hidden_layer_weights_initializer=model_config_dict[
        "fcnet_kernel_initializer"
    ],
    hidden_layer_weights_initializer_config=model_config_dict[
        "fcnet_kernel_initializer_kwargs"
    ],
    hidden_layer_bias_initializer=model_config_dict[
        "fcnet_bias_initializer"
    ],
    hidden_layer_bias_initializer_config=model_config_dict[
        "fcnet_bias_initializer_kwargs"
    ],
    output_layer_dim=encoder_latent_dim,
    output_layer_activation=model_config_dict["fcnet_activation"],
```

Key configuration aspects:

- **Input dimensions**: Set to the aggregated observation shape
- **Hidden layers**: Configured from the model config (all but the last layer)
- **Output dimension**: The final layer size becomes the encoder's latent dimension
- **Initialization**: Weight and bias initializers are preserved from the base configuration

> **💡 Info Point - MLP Architecture**: The Multi-Layer Perceptron (MLP) encoder creates a fully-connected network that maps from flattened observations to a latent representation. This latent space serves as input to the policy and value heads.

---

## 📚 6. Q-Function Encoder Architecture

The Q-function in SAC requires both observation and action as input to estimate state-action values. The catalog handles this concatenation:

```77:83:src/training/catalogs/structured_observation_sac_catalog.py
# Encoder input for the Q-network contains state and action. We
# need to infer the shape for the input from the state and action
# spaces
aggregated_obs_shape = sum([space.shape[-1] for space in self.observation_space.spaces.values()])
input_space = gym.spaces.Box(
    -np.inf,
    np.inf,
    (aggregated_obs_shape + required_action_dim,),
    dtype=np.float32,
)
```

The process involves:

1. **Action dimension calculation** - Determine action space size
2. **Input space creation** - Combine observation and action dimensions
3. **Encoder configuration** - Build MLP for state-action value estimation

For different action space types:

```64:72:src/training/catalogs/structured_observation_sac_catalog.py
# Compute the required dimension for the action space.
if isinstance(self.action_space, gym.spaces.Box):
    required_action_dim = self.action_space.shape[0]
elif isinstance(self.action_space, gym.spaces.Discrete):
    # for discrete action spaces, we don't need to encode the action
    # because the Q-function will output a value for each action
    required_action_dim = 0
else:
    self._raise_unsupported_action_space_error()
```

> **💡 Info Point - Q-Function Input**: In continuous control (Box action spaces), the Q-function takes both state and action as input. For discrete actions, the Q-function outputs a value for each possible action, so actions aren't part of the input.

---

## 📚 7. Encoder Building Process

The final Q-function encoder is built with the computed configurations:

```85:96:src/training/catalogs/structured_observation_sac_catalog.py
self.qf_encoder_hiddens = self._model_config_dict["fcnet_hiddens"][:-1]
self.qf_encoder_activation = self._model_config_dict["fcnet_activation"]

self.qf_encoder_config = MLPEncoderConfig(
    input_dims=input_space.shape,
    hidden_layer_dims=self.qf_encoder_hiddens,
    hidden_layer_activation=self.qf_encoder_activation,
    output_layer_dim=self.latent_dims[0],
    output_layer_activation=self.qf_encoder_activation,
)

return self.qf_encoder_config.build(framework=framework)
```

This configuration creates an MLP that:
- Takes concatenated state-action input
- Uses the same hidden layer structure as the policy encoder
- Outputs to a latent dimension used by the Q-function head

> **💡 Info Point - Framework Agnostic**: The `framework` parameter allows the same catalog configuration to build encoders for different deep learning frameworks (TensorFlow, PyTorch), providing flexibility in implementation.

---

## 📚 8. Integration with Training Pipeline

The catalog integrates with the training pipeline through the RLModuleSpec in the training configuration:

```30:30:src/config_factory/config_basemodels/training_config_basemodel.py
sac_config.rl_module(rl_module_spec=RLModuleSpec(catalog_class=StructuredObservationSACCatalog))
```

This connection ensures that:
1. **SAC algorithm** uses our custom catalog for module creation
2. **Observation processing** follows our dictionary space handling
3. **Neural architectures** match our encoder configurations
4. **Training** proceeds with properly configured networks

> **💡 Info Point - Catalog Integration**: The catalog system provides a clean interface between algorithm logic and network architecture. Changing catalogs allows experimenting with different network designs without modifying the core training loop.

---

## 📚 9. Benefits of Custom Catalogs

This custom catalog implementation provides several advantages:

### **Observation Space Flexibility**
Handles complex dictionary observations that standard catalogs can't process, enabling training on realistic multi-sensor environments.

### **Architecture Customization**
Allows fine-tuned control over network architectures while maintaining compatibility with RLlib's training infrastructure.

### **Code Reusability**
The catalog can be reused across different experiments and environments with similar observation structures.

### **Maintainability**
Separates observation processing logic from algorithm implementation, making both easier to understand and modify.

### **Type Safety**
Validates observation space structure early, preventing runtime errors during training.

> **💡 Info Point - Production Benefits**: Custom catalogs are essential for deploying RL systems in real-world applications where observation spaces don't match standard assumptions. They provide the flexibility needed for practical implementation.

---

## 📚 10. Catalog System Architecture

The catalog system in RLlib follows a well-defined architecture:

```python
Algorithm → RLModuleSpec → Catalog → EncoderConfig → Neural Network
```

Each component has specific responsibilities:

- **Algorithm**: Defines the learning procedure (SAC, PPO, etc.)
- **RLModuleSpec**: Specifies which catalog to use
- **Catalog**: Defines how to build network components
- **EncoderConfig**: Contains architecture specifications
- **Neural Network**: The actual trainable model

This separation allows modular composition where different algorithms can use different catalogs, and different catalogs can be used with the same algorithm.

> **💡 Info Point - Modular Design**: The catalog system exemplifies good software engineering practices by separating concerns and providing clear interfaces between components. This design makes RLlib both powerful and maintainable.

---

## 📚 11. When to Create Custom Catalogs

Consider creating custom catalogs when:

- **Complex Observation Spaces**: Your environment produces structured, multi-modal, or hierarchical observations
- **Specialized Architectures**: You need specific network designs (CNNs for images, attention mechanisms, etc.)
- **Domain Knowledge**: You want to incorporate domain-specific architectural choices
- **Performance Optimization**: Standard architectures don't perform well on your specific problem

The `StructuredObservationSACCatalog` demonstrates how to handle the first case - complex observation spaces that require custom preprocessing and dimension handling.

> **💡 Info Point - Customization Decision**: Creating custom catalogs requires balancing flexibility with complexity. Start with standard catalogs and customize only when standard approaches prove insufficient for your specific use case.
