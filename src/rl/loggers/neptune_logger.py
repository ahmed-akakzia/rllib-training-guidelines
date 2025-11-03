import os
import time
from dataclasses import asdict
from typing import Any, Optional, Generator

import neptune
import numpy as np
import pandas as pd
from neptune.utils import stringify_unsupported
from ray.tune.experiment.trial import Trial
from ray.tune.logger import LoggerCallback
from ray.tune.result import TIMESTEPS_TOTAL



class NeptuneLogger(LoggerCallback):
    def __init__(self, *args, **kwargs):
        self.experiment_id = os.getenv("NEPTUNE_SYS_ID", "")
        self.project_name = "rllib-training"
        self.experiment_name = "sac-training"
        self.neptune_tags = []
    
    def setup(self, *args, **kwargs):
        self.run = neptune.init_run(
            with_id=self.experiment_id,
            project=self.project_name,
            name=self.experiment_name,
            tags=self.neptune_tags,
            capture_stderr=not self.experiment_id,
            capture_stdout=not self.experiment_id,
            mode="async",
        )

    def log_trial_start(self, trial: Trial) -> None:
        pass

    def log_trial_result(self, iteration: int, trial: Trial, result: dict) -> None:
        if result["training_iteration"] % 10 == 0:
            self.run.sync()
        
        filtered_result = {k: v for k, v in result.items() if k in ["env_runners", "learners", "evaluation"]}

        for name, value in flat_items(filtered_result):
            if isinstance(value, str):
                if len(value) > 100:
                    continue
                self.run[name].append(value=value, step=result.get(TIMESTEPS_TOTAL))
            elif np.isscalar(value) and not np.isnan(value):
                self.run[name].append(value=value, step=result.get(TIMESTEPS_TOTAL))
            else:
                continue

    def log_trial_end(self, trial: Trial, failed: bool = False) -> None:
        pass

    # """RLlib Neptune logger."""

    # global_config: Any  # to fill dynamically in run.py

    # def _init(self) -> None:
    #     if "NEPTUNE_API_TOKEN" not in os.environ:
    #         raise OSError(
    #             "Environment variable 'NEPTUNE_API_TOKEN' is needed to log "
    #             "experiments to Neptune. Please add your token as an env variable."
    #         )
    #     assert type(self.trial) == Trial, "Unexpected `None' Trial type"
    #     neptune_config = self.global_config.neptune_config
    #     project_name = neptune_config.get("neptune_project_name")
    #     experiment_name = self.trial.trial_id.split("_")[0]
    #     experiment_id = os.getenv("NEPTUNE_SYS_ID", "")
    #     neptune_tags = list(neptune_config["tags"])
    #     self.run = neptune.init_run(
    #         with_id=experiment_id,
    #         project=project_name,
    #         name=experiment_name,
    #         tags=neptune_tags,
    #         capture_stderr=not experiment_id,
    #         capture_stdout=not experiment_id,
    #         mode="async",
    #     )
    #     time.sleep(2)
    #     self.run["parameters"] = stringify_unsupported(
    #         flat_items_dict(asdict(self.global_config))
    #     )

    # def on_result(self, result: dict[str, Any]) -> None:
    #     if result["training_iteration"] % 10 == 0:
    #         self.run.sync()

    #     for name, value in flat_items(result):
    #         if isinstance(value, str):
    #             if len(value) > 100:
    #                 continue
    #             self.run[name].append(value=value, step=result.get(TIMESTEPS_TOTAL))
    #         elif np.isscalar(value) and not np.isnan(value):
    #             self.run[name].append(value=value, step=result.get(TIMESTEPS_TOTAL))
    #         else:
    #             continue

    # def close(self) -> None:
    #     self.run.stop()


def flat_items_dict(d: dict, sep: str = "/", prefix: Optional[str] = None) -> dict:
    """Like flat_items but returns a dict instead of a generator: flatten all values
    recursively inside a dict.


    Args:
        d: a potentially nested dict.
        sep: the separator used to build the key `f"{prefix}{sep}{key}` when flattening.
        prefix: the prefix used to build the key `f"{prefix}{sep}{key}` when flattening.

    Returns:
        flattened dict items.

    Examples:

        flat_items_dict({})
        {}

        flat_items_dict({1: 2, 2: 3})
        {1: 2, 2: 3}

        flat_items_dict({1: 2, 2: 3, "train": {"acc": 0.9}})
        {1: 2, 2: 3, 'train/acc': 0.9}

        flat_items_dict({1: 2, 2: 3, "train": {"acc": 0.9, "loss": 0.12}})
        {1: 2, 2: 3, 'train/acc': 0.9, 'train/loss': 0.12}

        g = flat_items_dict({1:2, "train": {"acc":0.9, "loss": {"kl":0.1, "l2":0.3}}})

        g
        {1: 2, 'train/acc':0.9, 'train/loss/kl':0.1, 'train/loss/l2':0.3}
    """
    ret = {}
    for k, v in d.items():
        k = k if prefix is None else f"{prefix}{sep}{k}"
        if isinstance(v, dict):
            ret.update(flat_items_dict(v, sep=sep, prefix=k))
        else:
            ret[k] = v
    return ret


def flat_items(d: dict, sep: str = "/", prefix: Optional[str] = None) -> Generator:
    """Like `dict.items()` but flatten all values recursively.

    Args:
        d: a potentially nested dict.
        sep: the separator used to build the key `f"{prefix}{sep}{key}` when flattening.
        prefix: the prefix used to build the key `f"{prefix}{sep}{key}` when flattening.

    Returns:
        Generator of the flattened dict items.

    Examples:

        flat_items({})
        <generator object flat_items at ...>

        list(flat_items({}))
        []

        list(flat_items({1: 2, 2: 3}))
        [(1, 2), (2, 3)]

        list(flat_items({1: 2, 2: 3, "train": {"acc": 0.9}}))
        [(1, 2), (2, 3), ('train/acc', 0.9)]

        list(flat_items({1: 2, 2: 3, "train": {"acc": 0.9, "loss": 0.12}}))
        [(1, 2), (2, 3), ('train/acc', 0.9), ('train/loss', 0.12)]

        g = flat_items({1: 2, "train": {"acc": 0.9, "loss": {"kl": 0.1, "l2": 0.3}}})
        list(g)
        [(1, 2), ('train/acc', 0.9), ('train/loss/kl', 0.1), ('train/loss/l2', 0.3)]
    """
    for k, v in d.items():
        k = k if prefix is None else f"{prefix}{sep}{k}"
        if isinstance(v, dict):
            yield from flat_items(v, sep=sep, prefix=k)
        else:
            yield (k, v)
