import pytest
from base_env.env import DiscreteGoalReach1D
from config_factory.config_basemodels.env_config_basemodel import EnvConfigBaseModel


class TestEnv:
    """Test the base environment."""

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

    def test_reset(self):
        """Test the reset method."""
        self.env.reset()
        assert self.env.state == 0
        assert self.env.goal <= self.env_config.default_max_goal_range
        assert self.env.step_count == 0
        assert self.env.max_steps == self.env_config.default_max_steps
        assert self.env.max_goal == self.env_config.default_max_goal_range

    @pytest.mark.parametrize("action", [0, 1, 2])
    def test_step__goal_not_reached(self, action: int):
        """Test the step method when the goal is not reached."""
        self.env.reset()
        self.env.goal = self.env_config.default_max_goal_range
        state, reward, done = self.env.step(action)
        if action == 0:
            assert state == 0
        elif action == 1:
            assert state == -1
        elif action == 2:
            assert state == 1
        assert reward == 0
        assert not done

    @pytest.mark.parametrize("action", [1, 2])
    def test_step__goal_reached(self, action: int):
        """Test the step method when the goal is reached."""
        self.env.reset()
        self.env.goal = -1 if action == 1 else 1
        state, reward, done = self.env.step(action)
        if action == 1:
            assert state == -1
        elif action == 2:
            assert state == 1
        assert reward == 1
        assert not done

    @pytest.mark.parametrize("action", [0, 1, 2])
    def test_step__max_steps_reached(self, action: int):
        """Test the step method when the maximum number of steps is reached."""
        self.env.reset()
        self.env.step_count = self.env.max_steps - 2
        _, _, done = self.env.step(action)
        assert not done
        _, _, done = self.env.step(action)
        assert done
