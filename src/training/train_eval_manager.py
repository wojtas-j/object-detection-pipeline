import hydra
from dataclasses import dataclass
from typing import TypeVar, Generic
from omegaconf import DictConfig

from src.log_config.logging_config import setup_logger
from src.training.model_trainer import ModelTrainer, YOLOTrainer
from src.training.model_evaluator import ModelEvaluator, YOLOEvaluator

log = setup_logger("src.training.train_eval_manager")
T = TypeVar('T', bound=ModelTrainer)
U = TypeVar('U', bound=ModelEvaluator)


@dataclass
class DualDictConfig:
    train: DictConfig
    eval: DictConfig


class TrainEvalManager(Generic[T]):
    """ Generic manager for training_and_evaluation and evaluation models. """
    def __init__(self, trainer: T | None = None, evaluator: U | None = None):
        """
        Initializer for model trainer and evaluator.

        :param trainer: Instance of a model trainer.
        :param evaluator: Instance of a model evaluator.
        """

        self.trainer = trainer
        self.evaluator = evaluator

    def train(self, cfg: DictConfig) -> None:
        """
        Train the model using specified configuration.

        :param cfg: Hydra configuration file.
        """

        log.info(f"Started {cfg.training.model} training: {cfg.training.name}")
        self.trainer.train_model(cfg)
        log.info(f"Finished {cfg.training.model} training: {cfg.training.name}")

    def evaluate(self, cfg: DictConfig) -> None:
        """
        Evaluate the model using specified configuration.

        :param cfg: Hydra configuration file.
        """

        log.info(f"Started {cfg.evaluation.model} evaluation: {cfg.evaluation.name}")
        self.evaluator.evaluate_model(cfg)
        log.info(f"Finished {cfg.evaluation.model} evaluation: {cfg.evaluation.name}")

    def run_manager(self, model_name: str, mode: int, cfg: DualDictConfig) -> None:
        """
        Process training and evaluation models from command line.

        :param model_name: Name of the model to train/evaluate.
        :param mode:
        :param cfg:
        :raises ValueError:
        """

        # Prepare proper classes
        trainers_map = {
            "yolo": YOLOTrainer,
        }
        evaluators_map = {
            "yolo": YOLOEvaluator,
        }

        if mode == 0: # training only
            if model_name not in trainers_map:
                log.error(f"Unknown trainer type: {model_name}")
                raise ValueError(f"Unknown trainer type: {model_name}")
            self.trainer = trainers_map.get(model_name)()
            self.evaluator = None
        if mode == 1: # evaluation only
            if model_name not in evaluators_map:
                log.error(f"Unknown evaluator type: {model_name}")
                raise ValueError(f"Unknown evaluator type: {model_name}")
            self.trainer = None
            self.evaluator = evaluators_map.get(model_name)()
        if mode == 2: # training and evaluation
            if model_name not in trainers_map or model_name not in evaluators_map:
                log.error(f"Unknown trainer or evaluator type: {model_name}")
                raise ValueError(f"Unknown trainer or evaluator type: {model_name}")
            self.trainer = trainers_map.get(model_name)()
            self.evaluator = evaluators_map.get(model_name)()
        else:
            log.error(f"Unknown mode: {mode}")
            raise ValueError(f"Unknown mode: {mode}")

        # Run
        try:
            if self.trainer is not None:
                self.train(cfg.train)
            if self.evaluator is not None:
                self.evaluate(cfg.eval)
        except Exception as e:
            log.error(f"Failed to execute: {model_name}: {e}")
            raise


@hydra.main(config_path="../../configs/training_and_evaluation", config_name="train_eval_manager", version_base=None)
def main(cfg: DictConfig) -> None:
    selected_model = cfg.manager.selected_model
    model = cfg.manager.models[selected_model]
    config = DualDictConfig(
        train = hydra.compose(config_name=f"train_{model.name}_{model.dataset_train}"),
        eval = hydra.compose(config_name=f"eval_{model.name}_{model.dataset_eval}")
    )
    manager = TrainEvalManager()
    manager.run_manager(selected_model, cfg.manager.mode, config)


if __name__ == '__main__':
    main()
