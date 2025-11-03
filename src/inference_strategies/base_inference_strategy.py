"""Base Inference Strategy class for reinforcement learning inference strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import numpy as np


class BaseInferenceStrategy(ABC):
    """Abstract base class for all inference strategies.

    This class defines the common interface that an agent (random, heuristic, RL ...) interacts with
    an environment in inference mode.
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
    ) -> None:
        """Initialize the inference strategy.

        Args:
            observation_space: The observation space of the environment
            action_space: The action space of the environment
        """
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def act(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        deterministic: bool = False,
    ) -> Union[int, np.ndarray, Dict[str, np.ndarray]]:
        """Select an action given an observation.

        Args:
            observation: Current observation from the environment
            deterministic: Whether to act deterministically (for evaluation)

        Returns:
            Action to take
        """
