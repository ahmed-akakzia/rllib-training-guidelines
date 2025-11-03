"""Run a rollout."""

import logging
import random

from base_env.env import DiscreteGoalReach1D
from config_factory import load_and_instantiate_hydra_config


def run_rollout():
    """Run a rollout."""
    config = load_and_instantiate_hydra_config()

    logger = logging.getLogger(__name__)

    logger.info("Starting rollout execution...")

    env = DiscreteGoalReach1D(config.environment)
    env.reset()
    if config.rollout.rollout_unit == "episodes":
        for _ in range(config.rollout.rollout_unit_value):
            env.reset()
            while True:
                action = random.choice(config.environment.valid_actions)
                _, _, done = env.step(action)
                if done:
                    break
        logger.info(
            "Rollout of %s episodes executed successfully", config.rollout.rollout_unit_value
        )
    elif config.rollout.rollout_unit == "steps":
        for _ in range(config.rollout.rollout_unit_value):
            env.reset()
            while True:
                action = random.choice(config.environment.valid_actions)
                _, _, done = env.step(action)
                if done:
                    break
        logger.info("Rollout of %s steps executed successfully", config.rollout.rollout_unit_value)


if __name__ == "__main__":
    run_rollout()
