
from typing import Any

from ray.rllib.connectors.connector_v2 import ConnectorV2
from training.connectors.env_to_module.transform_observation_dict_to_array import TransformObservationDictToArray


def make_custom_env_to_module_connector_pipeline(env: Any, spaces: Any, device: Any) -> list[ConnectorV2]:  # pylint: disable=unused-argument
    """Make a custom env-to-module connector pipeline.
        
    This function defines the custom env-to-module connectors that will be added on top of the 
    default env-to-module connectors.

    Args:
        env: The environment.
        spaces: The spaces.
        device: The device.

    Returns:
        A list of connectors.
    """
    return [TransformObservationDictToArray()]
