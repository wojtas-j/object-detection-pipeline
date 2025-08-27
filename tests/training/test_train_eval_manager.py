import pytest
from unittest.mock import patch
from omegaconf import DictConfig

from src.training.train_eval_manager import TrainEvalManager, DualDictConfig, main
from src.exceptions.exceptions import YamlConfigError
from src.training.model_trainer import YOLOTrainer
from src.training.model_evaluator import YOLOEvaluator


#region Fixtures
@pytest.fixture
def mock_config():
    """ Create a mock Hydra configuration for train_eval_manager. """
    return DictConfig({
        "manager": {
            "mode": 2,
            "selected_model": "yolo",
            "selected_dataset_train": "coco",
            "selected_dataset_eval": "bdd100k",
            "models": {
                "yolo": {
                    "name": "yolo",
                    "datasets_train": ["coco", "bdd100k"],
                    "datasets_eval": ["coco", "bdd100k"]
                },
                "faster-rcnn": {
                    "name": "faster-rcnn",
                    "datasets_train": ["coco", "bdd100k"],
                    "datasets_eval": ["coco", "bdd100k"]
                }
            }
        }
    })


@pytest.fixture
def mock_dual_config():
    """ Create a mock DualDictConfig for testing. """
    train_cfg = DictConfig({
        "training": {
            "name": "yolo_coco",
            "model": "yolov8n.pt",
            "data_config": "configs/datasets/coco.yaml",
            "epochs": 10,
            "batch": 16,
            "imgsz": 640,
            "device": 0,
            "patience": 5,
            "optimizer": "Adam",
            "lr0": 0.001
        }
    })
    eval_cfg = DictConfig({
        "evaluation": {
            "name": "yolo_bdd100k",
            "data_config": "configs/datasets/bdd100k.yaml",
            "model_path": "runs/train/yolo_coco/weights/best.pt"
        }
    })
    return DualDictConfig(train=train_cfg, eval=eval_cfg)


@pytest.fixture
def train_eval_manager():
    """ Create a TrainEvalManager instance with mocked trainer and evaluator. """
    trainer = YOLOTrainer()
    evaluator = YOLOEvaluator()
    return TrainEvalManager(trainer=trainer, evaluator=evaluator)
#endregion

def test_train_eval_manager_init():
    """ Test TrainEvalManager initialization. """
    manager = TrainEvalManager(trainer=YOLOTrainer(), evaluator=YOLOEvaluator())

    # Verify results
    assert isinstance(manager.trainer, YOLOTrainer), "Expected trainer to be YOLOTrainer"
    assert isinstance(manager.evaluator, YOLOEvaluator), "Expected evaluator to be YOLOEvaluator"


def test_train_eval_manager_train(train_eval_manager, mock_dual_config):
    """ Test train method of TrainEvalManager. """
    with patch.object(YOLOTrainer, "train_model") as mock_train:
        train_eval_manager.train(mock_dual_config.train)
        mock_train.assert_called_once_with(mock_dual_config.train)

        # Verify results
        assert mock_train.call_count == 1

def test_train_eval_manager_evaluate(train_eval_manager, mock_dual_config):
    """ Test evaluate method of TrainEvalManager. """
    with patch.object(YOLOEvaluator, "evaluate_model") as mock_evaluate:
        train_eval_manager.evaluate(mock_dual_config.eval)
        mock_evaluate.assert_called_once_with(mock_dual_config.eval)

        # Verify results
        assert mock_evaluate.call_count == 1


def test_run_manager_train_only(train_eval_manager, mock_dual_config):
    """ Test run_manager in train-only mode (mode=0). """
    with patch.object(YOLOTrainer, "train_model") as mock_train:
        train_eval_manager.run_manager(model_name="yolo", mode=0, cfg=mock_dual_config)
        mock_train.assert_called_once_with(mock_dual_config.train)

        # Verify results
        assert train_eval_manager.trainer is not None, "Expected trainer to be set"
        assert train_eval_manager.evaluator is None, "Expected evaluator to be None"


def test_run_manager_eval_only(train_eval_manager, mock_dual_config):
    """ Test run_manager in eval-only mode (mode=1). """
    with patch.object(YOLOEvaluator, "evaluate_model") as mock_evaluate:
        train_eval_manager.run_manager(model_name="yolo", mode=1, cfg=mock_dual_config)
        mock_evaluate.assert_called_once_with(mock_dual_config.eval)

        # Verify results
        assert train_eval_manager.trainer is None, "Expected trainer to be None"
        assert train_eval_manager.evaluator is not None, "Expected evaluator to be set"


def test_run_manager_train_and_eval(train_eval_manager, mock_dual_config):
    """ Test run_manager in train-and-eval mode (mode=2). """
    with patch.object(YOLOTrainer, "train_model") as mock_train, patch.object(YOLOEvaluator, "evaluate_model") as mock_evaluate:
        train_eval_manager.run_manager(model_name="yolo", mode=2, cfg=mock_dual_config)
        mock_train.assert_called_once_with(mock_dual_config.train)
        mock_evaluate.assert_called_once_with(mock_dual_config.eval)

        # Verify results
        assert train_eval_manager.trainer is not None, "Expected trainer to be set"
        assert train_eval_manager.evaluator is not None, "Expected evaluator to be set"


def test_run_manager_invalid_model(train_eval_manager, mock_dual_config):
    """ Test run_manager with invalid model name. """
    # Verify results
    with pytest.raises(YamlConfigError, match="Unknown trainer type: invalid_model"):
        train_eval_manager.run_manager(model_name="invalid_model", mode=0, cfg=mock_dual_config)


def test_run_manager_invalid_mode(train_eval_manager, mock_dual_config):
    """ Test run_manager with invalid mode. """
    # Verify results
    with pytest.raises(YamlConfigError, match="Unknown mode: 3"):
        train_eval_manager.run_manager(model_name="yolo",  mode=3, cfg=mock_dual_config)


@patch("src.training.train_eval_manager.hydra.compose")
def test_main_valid_config(mock_compose, mock_config):
    """ Test main function with valid configuration. """
    mock_train_cfg = DictConfig({"training": {"name": "yolo_coco"}})
    mock_eval_cfg = DictConfig({"evaluation": {"name": "yolo_bdd100k"}})
    mock_compose.side_effect = [mock_train_cfg, mock_eval_cfg]

    # Verify results
    with patch.object(TrainEvalManager, "run_manager") as mock_run_manager:
        main(mock_config)
        mock_compose.assert_any_call(config_name="train_yolo_coco")
        mock_compose.assert_any_call(config_name="eval_yolo_bdd100k")
        mock_run_manager.assert_called_once_with(
            "yolo",
            mock_config.manager.mode,
            DualDictConfig(train=mock_train_cfg, eval=mock_eval_cfg)
        )


def test_main_invalid_model(mock_config):
    """ Test main function with invalid selected_model. """
    mock_config.manager.selected_model = "invalid_model"
    # Verify results
    with pytest.raises(YamlConfigError, match="Invalid or missing selected_model: invalid_model"):
        main(mock_config)


def test_main_invalid_dataset_train(mock_config):
    """ Test main function with invalid selected_dataset_train. """
    mock_config.manager.selected_dataset_train = "invalid_dataset"
    # Verify results
    with pytest.raises(YamlConfigError, match="Invalid dataset for training: invalid_dataset"):
        main(mock_config)


def test_main_invalid_dataset_eval(mock_config):
    """ Test main function with invalid selected_dataset_eval. """
    mock_config.manager.mode = 1
    mock_config.manager.selected_dataset_eval = "invalid_dataset"

    # Verify results
    with pytest.raises(YamlConfigError, match="Invalid dataset for evaluation: invalid_dataset"):
        main(mock_config)


def test_main_missing_dataset_train(mock_config):
    """ Test main function with missing selected_dataset_train. """
    mock_config.manager.selected_dataset_train = None

    # Verify results
    with pytest.raises(YamlConfigError, match="Invalid or missing selected_dataset_train: None"):
        main(mock_config)


def test_main_missing_dataset_eval(mock_config):
    """ Test main function with missing selected_dataset_eval. """
    mock_config.manager.mode = 1
    mock_config.manager.selected_dataset_eval = None

    # Verify results
    with pytest.raises(YamlConfigError, match="Invalid or missing selected_dataset_eval: None"):
        main(mock_config)
