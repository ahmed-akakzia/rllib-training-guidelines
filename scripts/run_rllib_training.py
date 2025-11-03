import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import gymnasium_robotics
from gymnasium_robotics.envs.fetch.pick_and_place import MujocoFetchPickAndPlaceEnv
from gymnasium_robotics.envs.fetch.reach import MujocoFetchReachEnv
import ray
from ray.tune import Tuner
from ray.rllib.algorithms.sac import SACConfig
from ray import tune
from rl.loggers.neptune_logger import NeptuneLogger
from wrapper.dict_to_array_obs_wrapper import DictToArrayObsWrapper
from ray.rllib.connectors.env_to_module.mean_std_filter import MeanStdFilter

fetch_class = MujocoFetchReachEnv


def env_creator(config):
    env = TimeLimit(DictToArrayObsWrapper(fetch_class(reward_type="dense")), max_episode_steps=50)
    return env

tune.register_env("CustomEnv", env_creator)

config = (
    SACConfig()
    .environment("CustomEnv")
    .env_runners(
        num_env_runners=1,
        rollout_fragment_length=50,
        env_to_module_connector=lambda env, spaces, device: (
                    MeanStdFilter()
                ), 
        # add_default_connectors_to_env_to_module_pipeline=False
    )
    .training(
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
ray.init(ignore_reinit_error=True, local_mode=False)

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
