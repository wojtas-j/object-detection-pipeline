from abc import ABC, abstractmethod
from omegaconf import DictConfig

from src.log_config.logging_config import setup_logger

log = setup_logger(__name__)


#region ModelTrainer
class ModelTrainer(ABC):
    """ Abstract class for training_and_evaluation different models. """

    @abstractmethod
    def modify_model(self, cfg: DictConfig) -> None:
        """
        Modify selected model.

        :param cfg: Hydra configuration file.
        """
        pass

    @abstractmethod
    def train_model(self, cfg: DictConfig) -> None:
        """
        Train a model using specified configuration file on specified dataset.

        :param cfg: Hydra configuration file.
        """
        pass

    @abstractmethod
    def log_final_metrics(self, cfg: DictConfig) -> None:
        """
        Log final metrics after training_and_evaluation and evaluation.

        :param cfg: Hydra configuration file.
        """
        pass

#endregion

#region YOLOTrainer
class YOLOTrainer(ModelTrainer):

    def modify_model(self, cfg: DictConfig) -> None:
        """
        Evaluate a model using specified configuration file on specified dataset.

        :param cfg: Hydra configuration file.
        """
        log.info(f"Modifying model {cfg.training.model}")

    def train_model(self, cfg: DictConfig) -> None:
        """
        Train a model using specified configuration file on specified dataset.

        :param cfg: Configuration file to evaluate.
        """
        log.info(f"Training model {cfg.training.name}")


    def log_final_metrics(self, cfg: DictConfig) -> None:
        """
        Log final metrics after training_and_evaluation and evaluation.

        :param cfg: Configuration file to evaluate.
        """
        log.info(f"Logging final metrics after training {cfg.training.name}")
#endregion
