import pytest
import torch
import re
import numpy as np
from typing import Any
from pathlib import Path
from unittest.mock import MagicMock, patch
from omegaconf import DictConfig
from ultralytics.utils.metrics import DetMetrics

from src.exceptions.exceptions import EvaluationError, YamlConfigError, MetricsLoggingError
from src.training.model_evaluator import YOLOEvaluator


#region Fixtures
@pytest.fixture
def sample_yolo_cfg(tmp_path):
    """ Create sample DictConfig for YOLO evaluation testing, based on eval_yolo_bdd100k.yaml. """
    data_yaml = tmp_path / "bdd100k_yolo.yaml"
    data_yaml.write_text(
        """
        path: bdd100k/
        train: images/train
        val: images/val
        nc: 7
        names: ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck']
        """
    )
    return DictConfig({
        "evaluation": {
            "base_model": "yolo11s.pt",
            "name": "bdd100k_evaluation",
            "project": str(tmp_path / "runs/yolo/evaluation"),
            "data": str(data_yaml),
            "dataset": "bdd100k",
            "model_path": str(tmp_path / "runs/yolo/train/coco_training/weights/best.pt"),
            "eval_videos": False,
            "videos_path": str(tmp_path / "videos"),
            "imgsz": 640,
            "batch": 16,
            "videos_batch": 1,
            "device": 0,
            "workers": 12,
            "conf": 0.25,
            "iou": 0.6
        }
    })


@pytest.fixture
def sample_yolo_det_metrics():
    """ Create sample DetMetrics for testing. """
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
    """ Provide a temporary directory for testing file operations. """
    return tmp_path


@pytest.fixture
def mock_yolo_model():
    """ Mock YOLO model for testing. """
    model = MagicMock()
    result = MagicMock()
    result.boxes = [MagicMock(xyxy=torch.tensor([[0, 0, 100, 100]]), conf=torch.tensor([0.9]), cls=torch.tensor([0]))]
    model.predict.return_value = [result]
    model.val.return_value = MagicMock(spec=DetMetrics)
    return model
#endregion

#region YOLOEvaluator
# Tests for YOLOEvaluator.evaluate_model
@patch("torch.cuda.is_available", return_value=False)
def test_evaluate_model_no_cuda(_mock_cuda, sample_yolo_cfg):
    """ Test evaluate_model raises EvaluationError when CUDA is not available. """
    sample_yolo_cfg.evaluation.device = 0
    evaluator = YOLOEvaluator()

    # Verify results
    with pytest.raises(EvaluationError, match="CUDA is not available"):
        evaluator.evaluate_model(sample_yolo_cfg)


@patch("src.training.model_evaluator.YOLO")
@patch("src.datasets.datasets_utils.path_exists", return_value=False)
def test_evaluate_model_invalid_model_path(_mock_path_exists, _mock_yolo, sample_yolo_cfg):
    """ Test evaluate_model raises YamlConfigError for invalid model path. """
    sample_yolo_cfg.evaluation.model_path = "invalid.pt"
    evaluator = YOLOEvaluator()

    # Verify results
    with pytest.raises(YamlConfigError, match="Model file not found: invalid.pt"):
        evaluator.evaluate_model(sample_yolo_cfg)


@patch("src.training.model_evaluator.YOLO")
@patch("src.training.model_evaluator.path_exists", return_value=True)
def test_evaluate_model_valid_images(_mock_path_exists, mock_yolo, sample_yolo_cfg, sample_yolo_det_metrics):
    """ Test evaluate_model with valid image evaluation. """
    mock_yolo.return_value.val.return_value = sample_yolo_det_metrics
    evaluator = YOLOEvaluator()
    evaluator.evaluate_model(sample_yolo_cfg)


    # Verify results
    mock_yolo.assert_called_once_with(sample_yolo_cfg.evaluation.model_path)
    mock_yolo.return_value.val.assert_called_once()


@patch("src.training.model_evaluator.YOLO")
@patch("src.training.model_evaluator.path_exists", return_value=False)
def test_evaluate_model_invalid_videos_path(mock_path_exists, _mock_yolo, sample_yolo_cfg):
    """ Test evaluate_model raises YamlConfigError for invalid videos path. """
    def path_exists_side_effect(path):
        if "best.pt" in str(path):
            return True
        return False

    mock_path_exists.side_effect = path_exists_side_effect
    sample_yolo_cfg.evaluation.eval_videos = True
    sample_yolo_cfg.evaluation.videos_path = "invalid_videos"
    evaluator = YOLOEvaluator()

    # Verify results
    with pytest.raises(YamlConfigError, match="Videos file not found: invalid_videos"):
        evaluator.evaluate_model(sample_yolo_cfg)


@patch("src.training.model_evaluator.YOLO")
@patch("src.training.model_evaluator.path_exists", return_value=True)
def test_evaluate_model_no_videos(_mock_path_exists, _mock_yolo, sample_yolo_cfg, tmp_path):
    """ Test evaluate_model raises YamlConfigError when no videos are found. """
    sample_yolo_cfg.evaluation.eval_videos = True
    videos_dir = tmp_path / "videos"
    sample_yolo_cfg.evaluation.videos_path = str(videos_dir)
    videos_dir.mkdir()
    evaluator = YOLOEvaluator()

    # Verify results
    with pytest.raises(YamlConfigError, match=r"No videos found.*" + re.escape(str(videos_dir))):
        evaluator.evaluate_model(sample_yolo_cfg)


@patch("src.training.model_evaluator.YOLO")
@patch("src.training.model_evaluator.path_exists", return_value=True)
@patch("src.log_config.train_eval_metric_utils.find_directory_with_files")
def test_evaluate_model_valid_videos(_mock_find_dir, _mock_path_exists, mock_yolo, sample_yolo_cfg, sample_yolo_det_metrics, tmp_path):
    """ Test evaluate_model with valid video evaluation. """
    sample_yolo_cfg.evaluation.eval_videos = True
    sample_yolo_cfg.evaluation.videos_path = str(tmp_path / "videos")
    (tmp_path / "videos" / "test.mp4").mkdir(parents=True)
    (tmp_path / "videos" / "test.mp4").touch()
    mock_yolo.return_value.val.return_value = sample_yolo_det_metrics
    evaluator = YOLOEvaluator()
    evaluator.evaluate_model(sample_yolo_cfg)

    # Verify results
    mock_yolo.assert_called_once_with(sample_yolo_cfg.evaluation.model_path)
    mock_yolo.return_value.val.assert_called_once()


@patch("src.training.model_evaluator.YOLO")
@patch("src.training.model_evaluator.path_exists", return_value=True)
def test_evaluate_model_image_evaluation_error(_mock_path_exists, mock_yolo, sample_yolo_cfg):
    """ Test evaluate_model handles image evaluation error. """
    mock_yolo.return_value.val.side_effect = RuntimeError("Evaluation failed")
    evaluator = YOLOEvaluator()

    # Verify results
    with pytest.raises(RuntimeError, match="Evaluation failed"):
        evaluator.evaluate_model(sample_yolo_cfg)


# Tests for YOLOEvaluator.log_final_metrics
@patch("src.training.model_evaluator.save_yolo_metrics")
@patch("src.training.model_evaluator.predict_yolo_evaluation_image")
@patch("src.training.model_evaluator.find_directory_with_files")
@patch("omegaconf.OmegaConf.load")
def test_log_final_metrics_images(_mock_omegaconf_load, _mock_find_dir, _mock_predict_image, _mock_save_yolo_metrics, sample_yolo_cfg, sample_yolo_det_metrics, tmp_path):
    """ Test log_final_metrics with valid DetMetrics for images. """
    mock_data_cfg = DictConfig({"path": "bdd100k", "val": "images/val"})
    _mock_omegaconf_load.return_value = mock_data_cfg
    _mock_find_dir.return_value = tmp_path / "datasets/bdd100k/images/val"
    evaluator = YOLOEvaluator()
    evaluator.log_final_metrics(sample_yolo_cfg, sample_yolo_det_metrics)

    # Verify results
    _mock_save_yolo_metrics.assert_called_once_with(
        sample_yolo_det_metrics,
        Path(sample_yolo_cfg.evaluation.project) / sample_yolo_cfg.evaluation.name,
        "yolo_image_evaluation_results.json"
    )
    _mock_predict_image.assert_called_once_with(sample_yolo_cfg, tmp_path / "datasets/bdd100k/images/val")


@patch("src.training.model_evaluator.save_yolo_metrics")
@patch("src.training.model_evaluator.predict_yolo_evaluation_video")
@patch("src.training.model_evaluator.find_directory_with_files")
def test_log_final_metrics_videos(_mock_find_dir, _mock_predict_video, _mock_save_yolo_metrics, sample_yolo_cfg, tmp_path):
    """ Test log_final_metrics with valid dict for videos. """
    _mock_find_dir.return_value = tmp_path / "videos"
    results = {
        "average_time_ms": 10.0,
        "average_flicker_rate": 0.1,
        "average_iou_consistency": 0.8
    }
    evaluator = YOLOEvaluator()
    evaluator.log_final_metrics(sample_yolo_cfg, results)

    # Verify results
    _mock_save_yolo_metrics.assert_called_once_with(
        results,
        Path(sample_yolo_cfg.evaluation.project) / sample_yolo_cfg.evaluation.name,
        "yolo_videos_evaluation_results.json"
    )
    _mock_predict_video.assert_called_once_with(sample_yolo_cfg, tmp_path / "videos")


def test_log_final_metrics_invalid_type(sample_yolo_cfg):
    """ Test log_final_metrics raises MetricsLoggingError for invalid results type. """
    evaluator = YOLOEvaluator()
    invalid_type: Any = object()

    # Verify results
    with pytest.raises(MetricsLoggingError, match="Invalid type of final metrics results"):
        evaluator.log_final_metrics(sample_yolo_cfg, invalid_type)


@patch("src.training.model_evaluator.find_directory_with_files")
@patch("omegaconf.OmegaConf.load")
def test_log_final_metrics_invalid_image_path(_mock_omegaconf_load, _mock_find_dir, sample_yolo_cfg, sample_yolo_det_metrics):
    """ Test log_final_metrics handles invalid image path. """
    mock_data_cfg = DictConfig({"path": "bdd100k", "val": "images/val"})
    _mock_omegaconf_load.return_value = mock_data_cfg
    _mock_find_dir.return_value = None
    evaluator = YOLOEvaluator()
    evaluator.log_final_metrics(sample_yolo_cfg, sample_yolo_det_metrics) # Should not raise, just log error


@patch("src.training.model_evaluator.find_directory_with_files")
def test_log_final_metrics_invalid_video_path(_mock_find_dir, sample_yolo_cfg):
    """ Test log_final_metrics handles invalid video path. """
    _mock_find_dir.return_value = None
    results = {"average_time_ms": 10.0}
    evaluator = YOLOEvaluator()
    evaluator.log_final_metrics(sample_yolo_cfg, results) # Should not raise, just log error


# Tests for YOLOEvaluator._get_eval_params
def test_get_yolo_eval_params(sample_yolo_cfg):
    """ Test _get_eval_params returns correct parameters. """
    evaluator = YOLOEvaluator()
    params = evaluator._get_eval_params(sample_yolo_cfg)
    expected = {
        "name": "bdd100k_evaluation",
        "project": sample_yolo_cfg.evaluation.project,
        "data": sample_yolo_cfg.evaluation.data,
        "imgsz": 640,
        "batch": 16,
        "device": 0,
        "workers": 12,
        "conf": 0.25,
        "iou": 0.6
    }

    # Verify results
    assert params == expected, f"Expected params {expected}, got {params}"


# Tests for YOLOEvaluator._evaluate_on_videos
@patch("src.training.model_evaluator.YOLO")
@patch("cv2.VideoCapture")
@patch("src.training.model_evaluator.detection_flicker_rate")
@patch("src.training.model_evaluator.iou_consistency")
def test_evaluate_on_videos_valid(_mock_iou_consistency, _mock_flicker_rate, _mock_video_capture, mock_yolo, sample_yolo_cfg, tmp_path):
    """ Test _evaluate_on_videos with valid video. """
    sample_yolo_cfg.evaluation.videos_path = str(tmp_path / "videos")
    video_path = tmp_path / "videos" / "test.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.touch()
    mock_cap = MagicMock()
    mock_cap.read.side_effect = [
        (True, np.zeros((480, 640, 3), dtype=np.uint8)),
        (False, None)
    ]
    _mock_video_capture.return_value = mock_cap

    mock_box = MagicMock()
    mock_box.xyxy = torch.tensor([[0, 0, 100, 100]])
    mock_box.conf = torch.tensor([0.9])
    mock_box.cls = torch.tensor([0])

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]

    mock_model = MagicMock()
    mock_model.return_value = [mock_result]
    mock_yolo.return_value = mock_model

    _mock_flicker_rate.return_value = 0.1
    _mock_iou_consistency.return_value = 0.8

    evaluator = YOLOEvaluator()
    results = evaluator._evaluate_on_videos(
        mock_yolo.return_value,
        sample_yolo_cfg,
        {"imgsz": 640, "conf": 0.25, "iou": 0.6, "device": 0}
    )

    # Verify results
    assert results["average_time_ms"] >= 0.0
    assert results["average_flicker_rate"] == 0.1
    assert results["average_iou_consistency"] == 0.8
    assert results["num_videos"] == 1
    assert results["frames_evaluated"] == 1


@patch("src.training.model_evaluator.YOLO")
@patch("cv2.VideoCapture")
@patch("src.training.model_evaluator.detection_flicker_rate")
@patch("src.training.model_evaluator.iou_consistency")
def test_evaluate_on_videos_empty_video(_mock_iou_consistency, _mock_flicker_rate, _mock_video_capture, mock_yolo, sample_yolo_cfg, tmp_path):
    """ Test _evaluate_on_videos with empty video. """
    sample_yolo_cfg.evaluation.videos_path = str(tmp_path / "videos")
    video_path = tmp_path / "videos" / "test.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.touch()
    mock_cap = MagicMock()
    mock_cap.read.side_effect = [(False, None)] # Empty video
    _mock_video_capture.return_value = mock_cap
    _mock_flicker_rate.return_value = 0.0
    _mock_iou_consistency.return_value = 0.0
    evaluator = YOLOEvaluator()
    results = evaluator._evaluate_on_videos(
        mock_yolo.return_value,
        sample_yolo_cfg,
        {"imgsz": 640, "conf": 0.25, "iou": 0.6, "device": 0}
    )

    # Verify results
    assert results["average_time_ms"] == 0.0
    assert results["average_flicker_rate"] == 0.0
    assert results["average_iou_consistency"] == 0.0
    assert results["num_videos"] == 1
    assert results["frames_evaluated"] == 0
#endregion