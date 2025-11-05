import gymnasium as gym 
from ray.rllib.core.models.base import Encoder
from ray.rllib.core.models.configs import MLPEncoderConfig, ModelConfig

import numpy as np 
from ray.rllib.algorithms.sac.sac_catalog import SACCatalog


class StructuredObservationSACCatalog(SACCatalog):
    """Structured observation SAC catalog.
    
    This catalog is used to enable RLlib's SAC to handle dictionary observation space."""
    
    @classmethod
    def _get_encoder_config(
        cls,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
    ) -> ModelConfig:
        """Get the encoder configuration."""
        if (
            isinstance(observation_space, gym.spaces.Dict)
        ):
            # Chech that all spaces inside the Dict are of Box types and have 1 shape
            for space in observation_space.spaces.values():
                assert isinstance(space, gym.spaces.Box)
                assert len(space.shape) == 1
            aggregated_shape = sum([space.shape[-1] for space in observation_space.spaces.values()])
            hidden_layer_dims = model_config_dict["fcnet_hiddens"][:-1]
            encoder_latent_dim = model_config_dict["fcnet_hiddens"][-1]
            encoder_config: ModelConfig = MLPEncoderConfig(
                input_dims=(aggregated_shape,),
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
                output_layer_weights_initializer=model_config_dict[
                    "fcnet_kernel_initializer"
                ],
                output_layer_weights_initializer_config=model_config_dict[
                    "fcnet_kernel_initializer_kwargs"
                ],
                output_layer_bias_initializer=model_config_dict[
                    "fcnet_bias_initializer"
                ],
                output_layer_bias_initializer_config=model_config_dict[
                    "fcnet_bias_initializer_kwargs"
                ],
            )
            return encoder_config
        else:
            return super()._get_encoder_config(observation_space, model_config_dict, action_space)
    
    def build_qf_encoder(self, framework: str) -> Encoder:
        """Build the Q-function encoder."""
        # Compute the required dimension for the action space.
        if isinstance(self.action_space, gym.spaces.Box):
            required_action_dim = self.action_space.shape[0]
        elif isinstance(self.action_space, gym.spaces.Discrete):
            # for discrete action spaces, we don't need to encode the action
            # because the Q-function will output a value for each action
            required_action_dim = 0
        else:
            self._raise_unsupported_action_space_error()

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