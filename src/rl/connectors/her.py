from collections import Counter
from typing import Any, List, Optional

import gymnasium as gym

from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.typing import EpisodeType


class HindsightExperienceReplay(ConnectorV2):
    """TODO"""

    def __init__(
        self,
        input_observation_space: Optional[gym.Space] = None,
        input_action_space: Optional[gym.Space] = None,
        *,
        intrinsic_reward_coeff: float = 1.0,
        **kwargs,
    ):
        """Initializes a CountBasedCuriosity instance.

        Args:
            intrinsic_reward_coeff: The weight with which to multiply the intrinsic
                reward before adding (and saving) it back to the main (extrinsic)
                reward of the episode at each timestep.
        """
        super().__init__(input_observation_space, input_action_space)

    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: Any,
        episodes: List[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        # Loop through all episodes and change the reward to
        # [reward + intrinsic reward]
        for sa_episode in self.single_agent_episode_iterator(
            episodes=episodes, agents_that_stepped_only=False
        ):
            # Loop through all observations, except the last one.
            observations = sa_episode.get_observations(slice(None, -1))
            # Get all respective extrinsic rewards.
            rewards = sa_episode.get_rewards()

            for i, (obs, rew) in enumerate(zip(observations, rewards)):
                # Add 1 to obs counter.
                obs = tuple(obs)
                # Compute the count-based intrinsic reward and add it to the extrinsic
                # reward.
                rew += 100
                # Store the new reward back to the episode (under the correct
                # timestep/index).
                sa_episode.set_rewards(new_data=rew, at_indices=i)

        return batch