"""Run a rollout."""

import logging
import random

from gymnasium.wrappers import TimeLimit
from gymnasium_robotics.envs.fetch.reach import MujocoFetchReachEnv

from config_factory import load_and_instantiate_hydra_config


def run_rollout():
    """Run a rollout."""
    config = load_and_instantiate_hydra_config()

    logger = logging.getLogger(__name__)

    logger.info("Starting rollout execution...")

    env = TimeLimit(
        MujocoFetchReachEnv(
            reward_type=config.environment.reward_type, 
            render_mode=config.environment.render_mode
        )
    , max_episode_steps=config.environment.max_episode_steps)

    if config.rollout.rollout_unit == "episodes":
        for _ in range(config.rollout.rollout_unit_value):
            env.reset()
            while True:
                action = env.action_space.sample()
                _, _, truncated, terminated, _ = env.step(action)
                env.render()
                if truncated or terminated:
                    break
    elif config.rollout.rollout_unit == "steps":
        env.reset()
        for _ in range(config.rollout.rollout_unit_value):
            action = env.action_space.sample()
            _, _, truncated, terminated, _ = env.step(action)
            env.render()
            if truncated or terminated:
                env.reset()

    logger.info("Successfully executed a rollout of %s %s", config.rollout.rollout_unit_value, config.rollout.rollout_unit)


if __name__ == "__main__":
    run_rollout()
