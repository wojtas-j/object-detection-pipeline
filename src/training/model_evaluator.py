from abc import ABC, abstractmethod
from omegaconf import DictConfig

from src.log_config.logging_config import setup_logger

log = setup_logger(name=__name__)


#region ModelEvaluator
class ModelEvaluator(ABC):
    """ Abstract class for evaluating different models. """

    @abstractmethod
    def modify_model(self, cfg: DictConfig) -> None:
        """
        Modify selected model.

        :param cfg: Hydra configuration file.
        """
        pass

    @abstractmethod
    def evaluate_model(self, cfg: DictConfig) -> None:
        """
        Evaluate a model using specified configuration file on specified dataset.

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

#region YOLOEvaluator
class YOLOEvaluator(ModelEvaluator):
    def modify_model(self, cfg: DictConfig) -> None:
        """
        Evaluate a model using specified configuration file on specified dataset.

        :param cfg: Hydra configuration file.
        """
        log.info(f"Modifying model {cfg.evaluation.name}")

    def evaluate_model(self, cfg: DictConfig) -> None:
        """
        Evaluate a model using specified configuration file on specified dataset.

        :param cfg: Hydra configuration file.
        """
        log.info(f"Evaluating model {cfg.evaluation.name}")

    def log_final_metrics(self, cfg: DictConfig) -> None:
        """
        Log final metrics after training_and_evaluation and evaluation.

        :param cfg: Hydra configuration file.
        """
        log.info(f"Logging final metrics after training_and_evaluation and evaluation.")
#endregion
