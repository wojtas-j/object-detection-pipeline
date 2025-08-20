import time
import torch
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from torch import nn
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

    def __init__(self):
        """ Initializer for model trainer and evaluator. """
        super().__init__()
        self.model = None

    def modify_model(self, cfg: DictConfig) -> None:
        """
        Evaluate a model using specified configuration file on specified dataset.

        :param cfg: Hydra configuration file.
        """
        log.info(f"Modifying model: {cfg.training.model}")
        self._replace_conv_with_depthwise_separable(self.model.model)

    def train_model(self, cfg: DictConfig) -> None:
        """
        Train a model using specified configuration file on specified dataset.

        :param cfg: Configuration file to evaluate.
        """
        if cfg.training.device == 0 and not torch.cuda.is_available():
            log.error("CUDA is not available, cannot train model")
            raise TrainingError("CUDA is not available")

        if cfg.training.transfer_learning:
            if cfg.training.modify_model:
                log.error("You should not modify model during transfer learning")
                raise YamlConfigError("You should not modify model during transfer learning")

            model_path = cfg.training.transfer_learning_model_path
            if model_path is None:
                log.error(f"Invalid or missing transfer_learning_model_path: {model_path}")
                raise YamlConfigError(f"Invalid or missing transfer_learning_model_path: {model_path}")
            if not path_exists(model_path):
                log.error(f"Model file not found: {model_path}")
                raise YamlConfigError(f"Model file not found: {model_path}")

            log.info(f"Transfer learning model {model_path}")
            self.model = YOLO(model_path)
        else:
            model_path = cfg.training.model
            if cfg.training.modify_model:
                self.model = YOLO(model_path.replace(".pt", ".yaml"))
                self.modify_model(cfg)
                log.info(f"Modified model: {model_path}")
            else:
                log.info(f"Standard model: {model_path}")
                self.model = YOLO(model_path)

        # Train
        train_params = self._get_train_params(cfg)
        start_time = time.time()
        try:
            results = self.model.train(**train_params)
        except (TrainingError, YamlConfigError, RuntimeError) as e:
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
        """
        Prepare training parameters.

        :param cfg: Hydra configuration.
        :return: Dictionary of training parameters.
        """
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

    def _replace_conv_with_depthwise_separable(self, module: nn.Module, parent_name: str="") -> None:
        """
        Recursively replace all Conv2d layers (except 1x1 convolutions and YOLO Detect head)
        with depthwise separable convolutions in a given model.
        This transformation preserves batch normalization (BN) and activation layers (e.g. SiLU)
        attached to the original convolution, and rewires them after the depthwise + pointwise blocks.

        :param module: The parent module to inspect and potentially modify.
        :param parent_name: Prefix for tracking nested module names during recursion
        """
        if module.__class__.__name__ == "Detect":
            return

        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name

            # Skip Conv2d 1x1
            if isinstance(child, nn.Conv2d) and child.kernel_size != (1, 1):
                # Preserve normalization and activation if they exist
                next_layers = []
                for attr in ['bn', 'act']:
                    if hasattr(module, f"{name}_{attr}"):
                        next_layers.append(getattr(module, f"{name}_{attr}"))

                # Create depthwise and pointwise convolutions
                def to_2tuple(x):
                    # if x is a single int -> return (x, x)
                    if isinstance(x, int):
                        return x, x
                    # if x is a tuple of tuples -> take the first element
                    if isinstance(x, tuple) and isinstance(x[0], tuple):
                        return x[0]
                    # if x is a tuple of ints -> return it as is
                    return x

                kernel_size = to_2tuple(child.kernel_size)
                stride = to_2tuple(child.stride)
                padding = to_2tuple(child.padding)

                depthwise = nn.Conv2d(
                    child.in_channels, child.in_channels,
                    kernel_size, stride,
                    padding, groups=child.in_channels, bias=False
                )
                pointwise = nn.Conv2d(
                    child.in_channels, child.out_channels, 1, 1, 0, bias=False
                )

                # New layer connects depthwise, pointwise, and preserved BN/SiLU
                new_layer = nn.Sequential(depthwise, pointwise, *next_layers)

                setattr(module, name, new_layer)
                log.info(f"Replaced {full_name}: Conv2d({child.in_channels}, {child.out_channels}, {child.kernel_size}) with Depthwise Separable Conv")
            else:
                # Recursively go into child modules
                self._replace_conv_with_depthwise_separable(child, full_name)
#endregion
