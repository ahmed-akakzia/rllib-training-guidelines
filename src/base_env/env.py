import random

from base_env.type_defs import ActionType, DoneType, RewardType, StateType
from config_factory.config_basemodels.env_config_basemodel import EnvConfigBaseModel


class DiscreteGoalReach1D:
    """A simple discrete goal-reaching environment.

    This environment defines a discrete scenario where an agent needs to reach a goal position
    in a one-dimensional discrete space.

    Initially the agent starts at position 0 and the goal is randomly selected from the range
    [-max_goal, max_goal]. At each step, the agent can either stay still (0), move left (1) or
    right (2). Whenever the agent reaches the goal, it receives a positive reward of 1. The episode
    ends after the maximum number of steps is reached.

    Attributes:
        max_goal: The maximum goal position the agent can reach.
        max_steps: The maximum number of steps the agent can take.
        state: The current position of the agent.
        goal: The target position the agent must reach to win.
        step_count: The number of steps taken.
        _config: The environment configuration.
    """

    def __init__(self, env_config: EnvConfigBaseModel) -> None:
        self.max_goal: int = env_config.default_max_goal_range
        self.max_steps: int = env_config.default_max_steps
        self.state: int = env_config.default_start_state
        self._config = env_config

        random.seed(env_config.seed)
        self.goal: int = random.randint(-self.max_goal, self.max_goal)

        self.step_count: int = 0

    @property
    def n_actions(self) -> int:
        """The number of actions the agent can take."""
        return len(self._config.valid_actions)

    def reset(self) -> StateType:
        """Resets the environment to the initial state.

        This sets the agent to the initial state and randomly selects a new goal.
        """
        self.state = self._config.default_start_state
        self.goal = random.randint(-self.max_goal, self.max_goal)
        self.step_count = 0
        return self.state

    def step(self, action: ActionType) -> tuple[StateType, RewardType, DoneType]:
        """Takes one step in the environment.

        Args:
            action: The action to take (0 for left, 1 for right).

        Returns:
            A tuple of (next_state, reward, done).
        """
        if action not in self._config.valid_actions:
            raise ValueError(
                f"Invalid action {action}. Must be within {self._config.valid_actions}."
            )

        self.update_state(action)

        self.step_count += 1

        reward = self.compute_reward()

        done = self.is_done()

        return self.state, reward, done

    def update_state(self, action: ActionType) -> None:
        """Updates the state of the agent.

        Args:
            action: The action to take.
        """
        if action == 1:
            self.state = max(self.state - 1, -self.max_goal)
        elif action == 2:
            self.state = min(self.state + 1, self.max_goal)

    def compute_reward(self) -> RewardType:
        """Computes the reward for the current state.

        Returns:
            The reward for the current state.
        """
        if self.state == self.goal:
            return 1.0
        return 0.0

    def is_done(self) -> DoneType:
        """Checks if the episode is done.

        Returns:
            True if the episode is done, False otherwise.
        """
        return self.step_count >= self.max_steps

    def render(self) -> None:
        """Renders the environment state in the console."""
        display_width = 2 * self.max_goal
        line = ["-"] * display_width
        agent_pos = self.state + self.max_goal
        goal_pos = self.goal + self.max_goal
        line[agent_pos] = "A"
        line[goal_pos] = "G"
        print("".join(line))
