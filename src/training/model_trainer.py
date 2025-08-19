import time
import torch
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ultralytics import YOLO

from src.datasets.datasets_utils import path_exists
from src.exceptions.exceptions import TrainingError, YamlConfigError
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
        if cfg.training.device == 0 and not torch.cuda.is_available():
            raise TrainingError("CUDA is not available")

        if cfg.training.transfer_learning:
            model_path = cfg.training.transfer_learning_model_path
            if model_path is None:
                raise YamlConfigError(f"Invalid or missing transfer_learning_model_path: {model_path}")
            if not path_exists(model_path):
                raise YamlConfigError(f"Model file not found: {model_path}")
        else:
            model_path = cfg.training.model

        model = YOLO(model_path)
        train_params = self._get_train_params(cfg)
        start_time = time.time()
        try:
            results = model.train(**train_params)
        except (TrainingError, RuntimeError) as e:
            end_time = time.time()
            total_time = end_time - start_time
            log.error(f"Training failed after {total_time:.2f}: {e}")
            raise
        else:
            end_time = time.time()
            total_time = end_time - start_time
            log.info(f"Training {cfg.training.model} finished {cfg.training.name} in {total_time:.2f} seconds.")

    def log_final_metrics(self, cfg: DictConfig) -> None:
        """
        Log final metrics after training_and_evaluation and evaluation.

        :param cfg: Configuration file to evaluate.
        """
        log.info(f"Logging final metrics after training {cfg.training.name}")

    def _get_train_params(self, cfg: DictConfig) -> dict:
        return {
            "name": cfg.training.name,
            "project": cfg.training.project,
            "data": cfg.training.data,
            "epochs": cfg.training.epochs,
            "imgsz": cfg.training.imgsz,
            "batch": cfg.training.batch,
            "device": cfg.training.device,
            "workers": cfg.training.workers,
            "optimizer": cfg.training.optimizer,
            "save_period": cfg.training.save_period,
            "patience": cfg.training.patience,
            "amp": cfg.training.amp,
            "cos_lr": cfg.training.cos_lr,
            "lr0": cfg.training.lr0,
            "lrf": cfg.training.lrf,
            "warmup_epochs": cfg.training.warmup_epochs,
            "warmup_momentum": cfg.training.warmup_momentum,
            "warmup_bias_lr": cfg.training.warmup_bias_lr,
            "freeze": cfg.training.freeze,
            "mixup": cfg.training.mixup,
            "copy_paste": cfg.training.copy_paste,
            "auto_augment": cfg.training.auto_augment,
            "hsv_h": cfg.training.hsv_h,
            "hsv_s": cfg.training.hsv_s,
            "hsv_v": cfg.training.hsv_v,
            "translate": cfg.training.translate,
            "scale": cfg.training.scale,
            "fliplr": cfg.training.fliplr,
            "mosaic": cfg.training.mosaic,
            "close_mosaic": cfg.training.close_mosaic,
            "erasing": cfg.training.erasing,
            "save_txt": cfg.training.save_txt,
            "save_conf": cfg.training.save_conf,
            "save_crop": cfg.training.save_crop
        }
#endregion
