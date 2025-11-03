import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium_robotics.envs.fetch.reach import MujocoFetchReachEnv
import numpy as np 
import ray
from ray.rllib.algorithms.sac.sac_catalog import SACCatalog
from ray.rllib.connectors import env_to_module
from ray.rllib.core.models.base import Encoder
from ray.rllib.core.models.configs import MLPEncoderConfig, ModelConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.tune import Tuner
from ray.rllib.algorithms.sac import SACConfig
from ray import tune
from rl.connectors import make_custom_env_to_module_connector_pipeline
from rl.connectors.her import HindsightExperienceReplay
from rl.loggers.neptune_logger import NeptuneLogger
from wrapper.dict_to_array_obs_wrapper import DictToArrayObsWrapper
from ray.rllib.connectors.env_to_module.mean_std_filter import MeanStdFilter

fetch_class = MujocoFetchReachEnv

class StructuredObservationSACCatalog(SACCatalog):
    @classmethod
    def _get_encoder_config(
        cls,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
    ) -> ModelConfig:
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
        """Todo"""
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


def env_creator(config):
    env = TimeLimit(fetch_class(reward_type="sparse"), max_episode_steps=50)
    return env

tune.register_env("CustomEnv", env_creator)

config = (
    SACConfig()
    .environment("CustomEnv")
    .env_runners(
        num_env_runners=1,
        rollout_fragment_length=50,
        # env_to_module_connector=lambda env, spaces, device: (
        #             MeanStdFilter()
        #         ), 
        env_to_module_connector=make_custom_env_to_module_connector_pipeline
        # add_default_connectors_to_env_to_module_pipeline=False
    )
    .training(
        learner_connector=lambda *ags, **kw: HindsightExperienceReplay(),
        actor_lr=1e-3, # type: ignore 
        critic_lr=1e-3, # type: ignore 
        alpha_lr=1e-3, # type: ignore 
        initial_alpha=0.2, # type: ignore 
        gamma=0.98, # type: ignore 
        tau=0.005, # type: ignore 
        train_batch_size_per_learner=256, # type: ignore 
        target_network_update_freq=2, # type: ignore 
        num_steps_sampled_before_learning_starts=0, # type: ignore
        training_intensity=10, 
    )
    .rl_module(rl_module_spec=RLModuleSpec(catalog_class=StructuredObservationSACCatalog))
    .reporting(
        min_sample_timesteps_per_iteration=500, 
    )
    .evaluation(
        evaluation_num_env_runners=1, 
        evaluation_duration=1, 
        evaluation_interval=10,
        evaluation_duration_unit="episodes",
        evaluation_force_reset_envs_before_iteration=True, 
        evaluation_config={"explore": False}
    )
)
# ray.init(local_mode=True)
# algo = config.build_algo()

# training_steps = 100
# eval_freq = 1

# for i in range(training_steps):
#     algo.train()
#     if i % eval_freq == 0:
#         result = algo.evaluate()
#         print(result["env_runners"]["episode_return_mean"])
#     print(f"Training step {i} completed")

# algo.stop()

# Initialize Tuner 
ray.init(ignore_reinit_error=True, local_mode=True)

tuner = Tuner(
    trainable="SAC",
    param_space=config.to_dict(), 
    run_config=tune.RunConfig(
        name="sac-training", 
        stop={"training_iteration": 500}, 
        callbacks=[NeptuneLogger()], 
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=10,
            checkpoint_at_end=True,
        )
    ) 
)

tuner.fit()
ray.shutdown()
