from pathlib import Path

from config_factory.config_basemodels.training_config_basemodel import TrainingConfigBaseModel
import hydra
from config_factory.config_basemodels.config_basemodel import ConfigBaseModel
from hydra import compose, initialize_config_dir

HYDRA_VERSION = str(hydra.__version__)


def get_root_path() -> Path:
    """Get the root path of the project by looking for a marker folder."""
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "src").exists():
            return parent
    raise RuntimeError("Project root not found")


def load_and_instantiate_hydra_config(
    config_path: Path | None = None, config_name: str = "config"
) -> ConfigBaseModel | TrainingConfigBaseModel:
    """Load and instantiate a hydra config."""
    if config_path is None:
        config_path = get_root_path() / "configs"

    with initialize_config_dir(config_dir=str(config_path), version_base=HYDRA_VERSION):
        config = compose(config_name=config_name)
    return hydra.utils.instantiate(config)
