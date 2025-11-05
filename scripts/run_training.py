from config_factory import get_root_path, load_and_instantiate_hydra_config
from config_factory.config_basemodels.training_config_basemodel import TrainingConfigBaseModel
import ray 


def run_training():
    """Run training."""
    config_path = get_root_path() / "configs" / "training"

    config = load_and_instantiate_hydra_config(config_path=config_path, config_name="training_config")

    assert isinstance(config, TrainingConfigBaseModel)

    tuner = config.to_tuner()

    ray.init(local_mode=config.local_mode)

    tuner.fit()
    
    ray.shutdown()



if __name__ == "__main__":
    run_training()