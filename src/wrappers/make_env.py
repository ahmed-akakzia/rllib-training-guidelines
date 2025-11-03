"""Environment factory."""

from base_env.env import DiscreteGoalReach1D
from config_factory.config_basemodels.env_config_basemodel import EnvConfigBaseModel
from wrappers import GymWrapper


def make_env(  # pylint: disable=unused-argument
    env_config: EnvConfigBaseModel,
    **kwargs,
) -> GymWrapper:
    """Create and wrap a gymnasium environment.

    This factory function creates a new instance of the environment and wraps it in a GymWrapper. It
    is used mostly in RL frameworks to ensure lazy creation, enable parallelization and ensure
    reproducibility.
    """
    env = DiscreteGoalReach1D(env_config)
    wrapped_env = GymWrapper(env)
    return wrapped_env
