from config_factory.config_basemodels.env_config_basemodel import EnvConfigBaseModel
from wrappers import GymWrapper, make_env


class TestMakeEnv:
    """Test the make_env function."""

    def setup_method(self):
        """Setup the attributes needed by the test class."""
        self.env_config = EnvConfigBaseModel(
            default_max_goal_range=5,
            default_max_steps=50,
            default_start_state=0,
            valid_actions=[0, 1, 2],
            seed=10,
        )

    def test_make_env(self):
        """Test the make_env function."""
        env = make_env(self.env_config)
        assert isinstance(env, GymWrapper)
