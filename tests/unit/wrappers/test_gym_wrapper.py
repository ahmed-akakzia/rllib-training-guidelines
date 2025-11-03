import pytest
from base_env.env import DiscreteGoalReach1D
from base_env.type_defs import ActionType, DoneType, RewardType
from config_factory.config_basemodels.env_config_basemodel import EnvConfigBaseModel
from wrappers import GymWrapper


class TestGymWrapper:
    """Test the gym wrapper."""

    def setup_method(self):
        """Setup the attributes needed by the test class."""
        self.env_config = EnvConfigBaseModel(
            default_max_goal_range=5,
            default_max_steps=50,
            default_start_state=0,
            valid_actions=[0, 1, 2],
            seed=10,
        )
        self.env = DiscreteGoalReach1D(self.env_config)
        self.env_wrapper = GymWrapper(self.env)

    def test_reset__base_env_is_reset(self):
        """Test the reset method."""
        self.env_wrapper.reset()
        assert self.env_wrapper.base_env.state == 0
        assert self.env_wrapper.base_env.goal <= self.env_config.default_max_goal_range
        assert self.env_wrapper.base_env.step_count == 0
        assert self.env_wrapper.base_env.max_steps == self.env_config.default_max_steps
        assert self.env_wrapper.base_env.max_goal == self.env_config.default_max_goal_range

    def test_reset__info_dict_is_returned(self):
        """Test the reset method."""
        _, info_dict = self.env_wrapper.reset()
        assert info_dict == {
            "goal": self.env_wrapper.base_env.goal,
            "step_count": self.env_wrapper.base_env.step_count,
            "max_steps": self.env_wrapper.base_env.max_steps,
            "max_goal": self.env_wrapper.base_env.max_goal,
        }

    @pytest.mark.parametrize("action", [0, 1, 2])
    def test_step__base_env_is_stepped(self, action: ActionType):
        """Test the step method."""
        _, info_dict = self.env_wrapper.reset()
        assert info_dict["step_count"] == 0
        _, _, _, _, info_dict = self.env_wrapper.step(action)
        assert info_dict["step_count"] == 1

    def test_step__gym_conventional_return_values_are_returned(self):
        """Test the step method."""
        self.env_wrapper.reset()
        action = 0
        observation, reward, terminated, truncated, info_dict = self.env_wrapper.step(action)
        assert observation in self.env_wrapper.observation_space
        assert isinstance(reward, RewardType)
        assert isinstance(terminated, DoneType)
        assert isinstance(truncated, DoneType)
