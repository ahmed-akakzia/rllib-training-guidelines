from typing import Any, Optional, Dict, List

import numpy as np 
import gymnasium as gym
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.env_to_module import MeanStdFilter
from ray.rllib.core import Columns
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import AgentID, EpisodeType
from ray.rllib.utils.filter import MeanStdFilter as _MeanStdFilter


class TransformObservationDictToArray(ConnectorV2):
    def __init__(
        self,
        input_observation_space: Optional[gym.Space] = None,
        input_action_space: Optional[gym.Space] = None,
        *,
        as_learner_connector: bool = False,
        **kwargs,
    ):
        """Todo"""
        super().__init__(
            input_observation_space=input_observation_space,
            input_action_space=input_action_space,
            **kwargs,
        )

        self._as_learner_connector = as_learner_connector

    @override(ConnectorV2)
    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: Dict[str, Any],
        episodes: List[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        # If "obs" already in data, early out.
        if Columns.OBS in batch:
            return batch
        for i, sa_episode in enumerate(
            self.single_agent_episode_iterator(
                episodes,
                # If Learner connector, get all episodes (for train batch).
                # If EnvToModule, get only those ongoing episodes that just had their
                # agent step (b/c those are the ones we need to compute actions for
                # next).
                agents_that_stepped_only=not self._as_learner_connector,
            )
        ):
            last_obs = sa_episode.get_observations(-1)
            concatenated_obs = np.concatenate([value for value in last_obs.values()])
            sa_episode.observations.pop(-1)
            sa_episode.observations.append(concatenated_obs)

        return batch


class CustomMeanStdFilter(MeanStdFilter):
    def __init__(
        self,
        *,
        multi_agent: bool = False,
        de_mean_to_zero: bool = True,
        de_std_to_one: bool = True,
        clip_by_value: Optional[float] = 10.0,
        update_stats: bool = True,
        **kwargs,
    ):
        """TODO"""
        self._multi_agent = multi_agent

        # We simply use the old MeanStdFilter until non-connector env_runner is fully
        # deprecated to avoid duplicate code
        self.de_mean_to_zero = de_mean_to_zero
        self.de_std_to_one = de_std_to_one
        self.clip_by_value = clip_by_value
        self._update_stats = update_stats

        self._filters: Optional[Dict[AgentID, _MeanStdFilter]] = None

        super().__init__(**kwargs)


def make_custom_env_to_module_connector_pipeline(env: Any, spaces: Any, device: Any) -> list[ConnectorV2]:  # pylint: disable=unused-argument
    """To Do 
    """
    # observation_space = env.observation_space
    # input_action_space = env.action_space
    # # Compute input obs space of the mean std filter
    # assert isinstance(observation_space, gym.spaces.Dict)
    # aggregated_obs_shape = sum([space.shape[-1] for space in observation_space.spaces.values()])
    # input_observation_space = gym.spaces.Box(
    #         low=-np.inf, 
    #         high= np.inf, 
    #         shape = (aggregated_obs_shape,)
    #     )
    # return [
    #     TransformObservationDictToArray(input_observation_space=input_observation_space, input_action_space=input_action_space), 
    #     CustomMeanStdFilter(input_observation_space=input_observation_space, input_action_space=input_action_space)
    # ]
    return [
        TransformObservationDictToArray()
    ]

