import hydra
from dataclasses import dataclass
from typing import TypeVar, Generic
from omegaconf import DictConfig

from src.exceptions.exceptions import TrainingError, YamlConfigError
from src.log_config.logging_config import setup_logger
from src.training.model_trainer import ModelTrainer, YOLOTrainer
from src.training.model_evaluator import ModelEvaluator, YOLOEvaluator

log = setup_logger("src.training.train_eval_manager")
T = TypeVar('T', bound=ModelTrainer)
U = TypeVar('U', bound=ModelEvaluator)


@dataclass
class DualDictConfig:
    """
    Wrapper for training and evaluation configurations.

    Attributes:
        train (DictConfig): Hydra config for training stage.
        eval (DictConfig): Hydra config for evaluation stage.
    """
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

        log.info(f"Starting training: {cfg.training.name}")
        self.trainer.train_model(cfg)

    def evaluate(self, cfg: DictConfig) -> None:
        """
        Evaluate the model using specified configuration.

        :param cfg: Hydra configuration file.
        """

        log.info(f"Started {cfg.evaluation.model} evaluation: {cfg.evaluation.name}")
        self.evaluator.evaluate_model(cfg)

    def run_manager(self, model_name: str, mode: int, cfg: DualDictConfig) -> None:
        """
        Process training and evaluation models from command line.

        :param model_name: Name of the model to train/evaluate.
        :param mode: Execution mode for the manager. (0 - training only, 1 - evaluation only, 2 - both).
        :param cfg: Hydra configuration file.
        :raises: TrainingError: Raised when model training fails due to CUDA/device issues or internal training errors.
        :raises: YamlConfigError: Raised when configuration is invalid.
        :raises: RuntimeError: Catch-all for unexpected runtime errors during training or evaluation.
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
                raise YamlConfigError(f"Unknown trainer type: {model_name}")
            self.trainer = trainers_map.get(model_name)()
            self.evaluator = None
        elif mode == 1: # evaluation only
            if model_name not in evaluators_map:
                log.error(f"Unknown evaluator type: {model_name}")
                raise YamlConfigError(f"Unknown evaluator type: {model_name}")
            self.trainer = None
            self.evaluator = evaluators_map.get(model_name)()
        elif mode == 2: # training and evaluation
            if model_name not in trainers_map or model_name not in evaluators_map:
                log.error(f"Unknown trainer or evaluator type: {model_name}")
                raise YamlConfigError(f"Unknown trainer or evaluator type: {model_name}")
            self.trainer = trainers_map.get(model_name)()
            self.evaluator = evaluators_map.get(model_name)()
        else:
            log.error(f"Unknown mode: {mode}")
            raise YamlConfigError(f"Unknown mode: {mode}")

        # Run
        try:
            if self.trainer is not None:
                self.train(cfg.train)
            if self.evaluator is not None:
                self.evaluate(cfg.eval)
        except (TrainingError, YamlConfigError, RuntimeError) as e:
            log.error(f"Failed to execute: {model_name}: {e}")
            raise


@hydra.main(config_path="../../configs/training_and_evaluation", config_name="train_eval_manager", version_base=None)
def main(cfg: DictConfig) -> None:
    selected_model = cfg.manager.selected_model
    selected_dataset_train = cfg.manager.selected_dataset_train
    selected_dataset_eval = cfg.manager.selected_dataset_eval

    if selected_model is None or selected_model not in cfg.manager.models:
        log.error(f"Invalid or missing selected_model: {selected_model}")
        raise YamlConfigError(f"Invalid or missing selected_model: {selected_model}")

    model = cfg.manager.models[selected_model]
    if cfg.manager.mode in (0, 2):
        if selected_dataset_train is None:
            log.error(f"Invalid or missing selected_dataset_train: {selected_dataset_train}")
            raise YamlConfigError(f"Invalid or missing selected_dataset_train: {selected_dataset_train}")
        if selected_dataset_train not in model.datasets_train:
            log.error(
                f"Invalid dataset for training: {selected_dataset_train}. \n"
                f"Available datasets: {model.datasets_train}")
            raise YamlConfigError(
                f"Invalid dataset for training: {selected_dataset_train}. \n"
                f"Available datasets: {model.datasets_train}"
            )

    if cfg.manager.mode in (1, 2):
        if selected_dataset_eval is None:
            log.error(f"Invalid or missing selected_dataset_eval: {selected_dataset_eval}")
            raise YamlConfigError(f"Invalid or missing selected_dataset_eval: {selected_dataset_eval}")
        if selected_dataset_eval not in model.datasets_eval:
            log.error(
                f"Invalid dataset for evaluation: {selected_dataset_eval}."
                f"Available datasets: {model.datasets_eval}"
            )
            raise YamlConfigError(
                f"Invalid dataset for evaluation: {selected_dataset_eval}."
                f"Available datasets: {model.datasets_eval}"
            )

    train_cfg = None
    eval_cfg = None
    if cfg.manager.mode in (0, 2) and selected_dataset_train is not None:
        train_cfg = hydra.compose(
            config_name=f"train_{model.name}_{selected_dataset_train}"
        )

    if cfg.manager.mode in (1, 2) and selected_dataset_eval is not None:
        eval_cfg = hydra.compose(
            config_name=f"eval_{model.name}_{selected_dataset_eval}"
        )

    config = DualDictConfig(train=train_cfg, eval=eval_cfg)
    manager = TrainEvalManager()
    manager.run_manager(selected_model, cfg.manager.mode, config)


if __name__ == '__main__':
    main()
