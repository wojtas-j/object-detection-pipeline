import pytest
import torch
from typing import Any
from omegaconf import DictConfig
from unittest.mock import MagicMock, patch
from ultralytics.utils.metrics import DetMetrics

from src.exceptions.exceptions import TrainingError, YamlConfigError, MetricsLoggingError
from src.training.model_trainer import YOLOTrainer

#region Fixtures
@pytest.fixture
def sample_yolo_cfg(tmp_path):
    """ Create sample DictConfig for testing YOLOTrainer. """
    return DictConfig({
        "training": {
            "model": "yolo11s.pt",
            "project": str(tmp_path / "runs/train"),
            "name": "test_train",
            "device": 0,
            "epochs": 1,
            "imgsz": 640,
            "batch": 16,
            "workers": 12,
            "optimizer": "SGD",
            "save_period": 1,
            "patience": 10,
            "amp": True,
            "cos_lr": False,
            "lr0": 0.01,
            "lrf": 0.01,
            "warmup_epochs": 3,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "freeze": None,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "auto_augment": "randaugment",
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "translate": 0.1,
            "scale": 0.5,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "close_mosaic": 0,
            "erasing": 0.4,
            "save_txt": False,
            "save_conf": False,
            "save_crop": False,
            "data": "data.yaml",
            "transfer_learning": False,
            "modify_model": False,
            "transfer_learning_model_path": None
        }
    })


@pytest.fixture
def sample_yolo_det_metrics():
    mock = MagicMock(spec=DetMetrics)
    mock.__class__ = DetMetrics
    mock.results_dict = {"train/loss": 0.5, "val/loss": 0.6, "metrics/mAP50": 0.8}
    mock.curves_results = {}
    mock.names = ["person", "car"]
    mock.nt_per_class = [100, 200]
    mock.ap_class_index = [0, 1]
    return mock


@pytest.fixture
def tmp_path(tmp_path):
    """ Create temporary path for testing YOLOTrainer. """
    return tmp_path
#endregion

#region YOLOTrainer
# Tests for YOLOTrainer.__init__
def test_yolo_trainer_init():
    """ Test YOLOTrainer initialization. """
    trainer = YOLOTrainer()

    # Verify results
    assert trainer.model is None, "Expected model to be None after initialization"


# Tests for YOLOTrainer.modify_model
@patch("src.training.model_trainer.YOLOTrainer._replace_conv_with_depthwise_separable")
def test_modify_model_calls_replace_conv(mock_replace_conv, sample_yolo_cfg):
    """ Test modify_model calls _replace_conv_with_depthwise_separable. """
    trainer = YOLOTrainer()
    trainer.model = MagicMock()
    sample_yolo_cfg.training.modify_model = True
    sample_yolo_cfg.training.model = "yolo11s.yaml"
    trainer.modify_model(sample_yolo_cfg)

    # Verify results
    mock_replace_conv.assert_called_once_with(trainer.model.model)


# Tests for YOLOTrainer.train_model
@patch("torch.cuda.is_available", return_value=False)
def test_train_model_no_cuda(_mock_cuda, sample_yolo_cfg):
    """ Test train_model raises TrainingError when CUDA is not available. """
    sample_yolo_cfg.training.device = 0
    trainer = YOLOTrainer()

    # Verify results
    with pytest.raises(TrainingError, match="CUDA is not available"):
        trainer.train_model(sample_yolo_cfg)


@patch("src.training.model_trainer.YOLO")
@patch("src.training.model_trainer.path_exists", return_value=True)
def test_train_model_transfer_learning_valid(_mock_path_exists, mock_yolo, sample_yolo_cfg, sample_yolo_det_metrics):
    """ Test train_model with valid transfer learning configuration. """
    sample_yolo_cfg.training.transfer_learning = True
    sample_yolo_cfg.training.transfer_learning_model_path = "yolo11s.pt"
    sample_yolo_cfg.training.modify_model = False
    mock_yolo.return_value.train.return_value = sample_yolo_det_metrics
    trainer = YOLOTrainer()
    trainer.train_model(sample_yolo_cfg)

    # Verify results
    mock_yolo.assert_called_once_with("yolo11s.pt")
    mock_yolo.return_value.train.assert_called_once()


def test_train_model_transfer_learning_modify_error(sample_yolo_cfg):
    """ Test train_model raises YamlConfigError when modify_model is True during transfer learning. """
    sample_yolo_cfg.training.transfer_learning = True
    sample_yolo_cfg.training.modify_model = True
    sample_yolo_cfg.training.transfer_learning_model_path = "yolo11s.pt"
    trainer = YOLOTrainer()

    # Verify results
    with pytest.raises(YamlConfigError, match="You should not modify model during transfer learning"):
        trainer.train_model(sample_yolo_cfg)


def test_train_model_transfer_learning_invalid_path(sample_yolo_cfg):
    """ Test train_model raises YamlConfigError for invalid transfer learning model path. """
    sample_yolo_cfg.training.transfer_learning = True
    sample_yolo_cfg.training.transfer_learning_model_path = "invalid.pt"
    trainer = YOLOTrainer()

    # Verify results
    with pytest.raises(YamlConfigError, match="Model file not found: invalid.pt"):
        trainer.train_model(sample_yolo_cfg)


@patch("src.training.model_trainer.YOLO")
@patch("src.training.model_trainer.YOLOTrainer.modify_model")
def test_train_model_standard_with_modify(mock_modify_model, mock_yolo, sample_yolo_cfg, sample_yolo_det_metrics):
    """ Test train_model with standard model and modification. """
    sample_yolo_cfg.training.modify_model = True
    sample_yolo_cfg.training.model = "yolo11s.pt"
    mock_yolo.return_value.train.return_value = sample_yolo_det_metrics
    trainer = YOLOTrainer()
    trainer.train_model(sample_yolo_cfg)
    mock_yolo.assert_called_once_with("yolo11s.yaml")

    # Verify results
    mock_modify_model.assert_called_once_with(sample_yolo_cfg)
    mock_yolo.return_value.train.assert_called_once()


@patch("src.training.model_trainer.YOLO")
def test_train_model_standard_no_modify(mock_yolo, sample_yolo_cfg, sample_yolo_det_metrics):
    """ Test train_model with standard model without modification. """
    sample_yolo_cfg.training.model = "yolo11s.pt"
    mock_yolo.return_value.train.return_value = sample_yolo_det_metrics
    trainer = YOLOTrainer()
    trainer.train_model(sample_yolo_cfg)

    # Verify results
    mock_yolo.assert_called_once_with("yolo11s.pt")
    mock_yolo.return_value.train.assert_called_once()


@patch("src.training.model_trainer.YOLO")
def test_train_model_training_error(mock_yolo, sample_yolo_cfg):
    """ Test train_model handles training error. """
    mock_yolo.return_value.train.side_effect = RuntimeError("Training failed")
    trainer = YOLOTrainer()

    # Verify results
    with pytest.raises(RuntimeError, match="Training failed"):
        trainer.train_model(sample_yolo_cfg)


# Tests for YOLOTrainer.log_final_metrics
@patch("src.training.model_trainer.save_yolo_metrics")
def test_log_final_metrics_valid(mock_save_yolo_metrics, sample_yolo_cfg, sample_yolo_det_metrics, tmp_path):
    """ Test log_final_metrics with valid DetMetrics. """
    sample_yolo_cfg.training.project = str(tmp_path / "runs")
    sample_yolo_cfg.training.name = "test_train"
    trainer = YOLOTrainer()
    trainer.log_final_metrics(sample_yolo_cfg, sample_yolo_det_metrics)

    # Verify results
    mock_save_yolo_metrics.assert_called_once_with(
        sample_yolo_det_metrics,
        tmp_path / "runs" / "test_train",
        "yolo_training_results.json"
    )


def test_log_final_metrics_invalid_type(sample_yolo_cfg):
    """ Test log_final_metrics raises MetricsLoggingError for invalid results type. """
    trainer = YOLOTrainer()
    invalid_results: Any = object()

    # Verify results
    with pytest.raises(MetricsLoggingError, match="Invalid type of final metrics results"):
        trainer.log_final_metrics(sample_yolo_cfg, invalid_results)


# Tests for YOLOTrainer._get_train_params
def test_get_train_params(sample_yolo_cfg, tmp_path):
    """ Test _get_train_params returns correct parameters. """
    trainer = YOLOTrainer()
    params = trainer._get_train_params(sample_yolo_cfg)
    expected = {
        "name": "test_train",
        "project": str(tmp_path / "runs/train"),
        "data": "data.yaml",
        "epochs": 1,
        "imgsz": 640,
        "batch": 16,
        "device": 0,
        "workers": 12,
        "optimizer": "SGD",
        "save_period": 1,
        "patience": 10,
        "amp": True,
        "cos_lr": False,
        "lr0": 0.01,
        "lrf": 0.01,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "freeze": None,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "auto_augment": "randaugment",
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "close_mosaic": 0,
        "erasing": 0.4,
        "save_txt": False,
        "save_conf": False,
        "save_crop": False
    }

    # Verify results
    assert params == expected, f"Expected params {expected}, got {params}"


# Tests for YOLOTrainer._replace_conv_with_depthwise_separable
def test_replace_conv_with_depthwise_separable_conv2d():
    """ Test _replace_conv_with_depthwise_separable replaces Conv2d with depthwise separable convolution. """
    trainer = YOLOTrainer()
    module = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.SiLU()
    )
    module.bn_0 = module[1] # Simulate named BN
    module.act_0 = module[2] # Simulate named activation
    trainer._replace_conv_with_depthwise_separable(module)
    assert isinstance(module[0], torch.nn.Sequential), "Expected Conv2d to be replaced with Sequential"
    assert isinstance(module[0][0], torch.nn.Conv2d), "Expected depthwise Conv2d"
    assert isinstance(module[0][1], torch.nn.Conv2d), "Expected pointwise Conv2d"
    assert module[0][0].groups == 3, "Expected depthwise Conv2d with groups equal to in_channels"
    assert module[0][1].kernel_size == (1, 1), "Expected pointwise Conv2d with 1x1 kernel"


def test_replace_conv_with_depthwise_separable_skip_1x1():
    """ Test _replace_conv_with_depthwise_separable skips 1x1 Conv2d. """
    trainer = YOLOTrainer()
    module = torch.nn.Sequential(torch.nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0))
    original_module = module[0]
    trainer._replace_conv_with_depthwise_separable(module)

    # Verify results
    assert module[0] is original_module, "Expected 1x1 Conv2d to be unchanged"


def test_replace_conv_with_depthwise_separable_detect():
    """ Test _replace_conv_with_depthwise_separable skips Detect module. """
    trainer = YOLOTrainer()
    module = MagicMock()
    module.__class__.__name__ = "Detect"
    trainer._replace_conv_with_depthwise_separable(module)

    # Verify results
    module.named_children.assert_not_called(), "Expected no modification for Detect module"
#endregion