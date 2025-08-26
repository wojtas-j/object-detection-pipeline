import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from omegaconf import DictConfig
from ultralytics.utils.metrics import DetMetrics, Metric

from src.log_config.train_eval_metric_utils import detection_flicker_rate, iou_consistency, _compute_iou, \
    _serialize_obj, find_directory_with_files, predict_yolo_evaluation_video, predict_yolo_evaluation_image, \
    save_yolo_metrics


#region Fixtures
@pytest.fixture
def sample_detections():
    """ Create sample detections for testing. """
    return [
        [[0, 0, 100, 100, 0.9, 0], [200, 200, 300, 300, 0.8, 1]], # Frame 1
        [[10, 10, 110, 110, 0.9, 0], [400, 400, 500, 500, 0.7, 1]], # Frame 2
        [[20, 20, 120, 120, 0.9, 0]] # Frame 3
    ]


@pytest.fixture
def empty_detections():
    """ Create empty detections list. """
    return []


@pytest.fixture
def single_frame_detections():
    """ Create detections for a single frame. """
    return [[[0, 0, 100, 100, 0.9, 0]]]


@pytest.fixture
def tmp_path(tmp_path):
    """ Temporary directory. """
    return tmp_path


@pytest.fixture
def mock_yolo_model():
    """ Mock YOLO model for testing predictions. """
    model = MagicMock()
    result = MagicMock()
    result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8) # Mock image
    result.path = "test_image.jpg"
    model.predict.return_value = [result]
    return model


@pytest.fixture
def sample_det_metrics():
    """ Create sample DetMetrics for testing using MagicMock. """
    det_metrics = MagicMock(spec=DetMetrics)
    det_metrics.results_dict = {
        "train/loss": 0.5,
        "val/loss": 0.6,
        "metrics/mAP50": 0.8
    }
    det_metrics.curves_results = [[0.1, 0.2], [0.3, 0.4]]
    det_metrics.names = ["person", "car"]
    det_metrics.nt_per_class = np.array([100, 200])
    det_metrics.ap_class_index = np.array([0, 1])
    return det_metrics


@pytest.fixture
def sample_metric():
    """ Create sample Metric for testing. """
    metric = Metric()
    metric.__dict__ = {"value": 0.9, "name": "precision"}
    return metric


@pytest.fixture
def sample_cfg():
    """ Create sample DictConfig for testing. """
    return DictConfig({
        "evaluation": {
            "model_path": "yolov8n.pt",
            "project": "runs/eval",
            "name": "test_eval",
            "imgsz": 640,
            "conf": 0.5
        }
    })
#endregion

#region save_yolo_metrics tests
def test_save_yolo_metrics_valid(tmp_path, sample_det_metrics):
    """ Test save_yolo_metrics with valid DetMetrics. """
    path_to_save = tmp_path / "metrics"
    file_name = "results.json"

    save_yolo_metrics(sample_det_metrics, path_to_save, file_name)
    output_path = path_to_save / file_name

    # Verify results
    assert output_path.exists(), f"Expected JSON file at {output_path}"
    with open(output_path, "r") as f:
        data = json.load(f)
    assert data["results_dict"]["train/loss"] == 0.5
    assert data["names"] == ["person", "car"]
    assert data["nt_per_class"] == [100, 200]


@patch("src.log_config.train_eval_metric_utils.log")
@patch("pathlib.Path.mkdir")
def test_save_yolo_metrics_invalid_path_logs(mock_mkdir, mock_log, tmp_path, sample_det_metrics):
    """Test save_yolo_metrics logs error when mkdir fails."""
    mock_mkdir.side_effect = OSError("Permission denied")
    save_yolo_metrics(sample_det_metrics, tmp_path / "some_path", "results.json")
    mock_log.error.assert_called_once()

    # Verify results
    assert "Failed to save YOLO metrics" in mock_log.error.call_args[0][0]


def test_save_yolo_metrics_empty_results(tmp_path):
    """ Test save_yolo_metrics with empty results. """
    path_to_save = tmp_path / "metrics"
    file_name = "empty_results.json"
    save_yolo_metrics({}, path_to_save, file_name)
    output_path = path_to_save / file_name

    # Verify results
    assert output_path.exists(), f"Expected JSON file at {output_path}"
    with open(output_path, "r") as f:
        data = json.load(f)
    assert data == {}, f"Expected empty dict, got {data}"
#endregion

#region predict_yolo_evaluation_image tests
@patch("src.log_config.train_eval_metric_utils.YOLO")
@patch("src.log_config.train_eval_metric_utils.cv2.imwrite")
def test_predict_yolo_evaluation_image_valid(mock_imwrite, mock_yolo, tmp_path, sample_cfg, mock_yolo_model):
    """ Test predict_yolo_evaluation_image with valid input. """
    model = mock_yolo_model()
    mock_yolo.return_value = model

    image_path = tmp_path / "images" / "test.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.touch()

    predict_yolo_evaluation_image(sample_cfg, tmp_path / "images")


    predict_directory_to = Path(sample_cfg.evaluation.project) / sample_cfg.evaluation.name
    output_folder = predict_directory_to / f"{sample_cfg.evaluation.name}_predict"
    output_path = output_folder / f"{Path(model.predict()[0].path).stem}_predict.jpg"

    # Verify results
    mock_imwrite.assert_called_once_with(str(output_path), model.predict()[0].plot())


def test_predict_yolo_evaluation_image_no_images(tmp_path, sample_cfg):
    """ Test predict_yolo_evaluation_image with no images in folder. """
    # Verify results
    with pytest.raises(IndexError):
        predict_yolo_evaluation_image(sample_cfg, tmp_path / "images")


@patch("src.log_config.train_eval_metric_utils.YOLO")
def test_predict_yolo_evaluation_image_invalid_model(mock_yolo, tmp_path, sample_cfg):
    """ Test predict_yolo_evaluation_image with invalid model path. """
    mock_yolo.side_effect = Exception("Invalid model")
    image_path = tmp_path / "images" / "test.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.touch()

    # Verify results
    predict_yolo_evaluation_image(sample_cfg, tmp_path / "images")
#endregion

#region predict_yolo_evaluation_video tests
@patch("src.log_config.train_eval_metric_utils.YOLO")
def test_predict_yolo_evaluation_video_valid(mock_yolo, tmp_path, sample_cfg, mock_yolo_model):
    """ Test predict_yolo_evaluation_video with valid input. """
    mock_yolo.return_value = mock_yolo_model()
    video_path = tmp_path / "videos" / "test.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.touch()

    predict_yolo_evaluation_video(sample_cfg, tmp_path / "videos")

    # Verify results
    mock_yolo.return_value.predict.assert_called_once()


def test_predict_yolo_evaluation_video_no_videos(tmp_path, sample_cfg):
    """ Test predict_yolo_evaluation_video with no videos in folder. """
    # Verify results
    with pytest.raises(IndexError):
        predict_yolo_evaluation_video(sample_cfg, tmp_path / "videos")


@patch("src.log_config.train_eval_metric_utils.YOLO")
def test_predict_yolo_evaluation_video_invalid_model(mock_yolo, tmp_path, sample_cfg):
    """ Test predict_yolo_evaluation_video with invalid model path. """
    mock_yolo.side_effect = Exception("Invalid model")
    video_path = tmp_path / "videos" / "test.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.touch()

    # Verify results
    predict_yolo_evaluation_video(sample_cfg, tmp_path / "videos")
#endregion

#region find_directory_with_files tests
def test_find_directory_with_files_base_folder(tmp_path):
    """ Test find_directory_with_files with files in base folder. """
    image_path = tmp_path / "test.jpg"
    image_path.touch()

    result = find_directory_with_files(tmp_path, [".jpg", ".png"])

    # Verify results
    assert result == tmp_path, f"Expected {tmp_path}, got {result}"


def test_find_directory_with_files_subfolder(tmp_path):
    """ Test find_directory_with_files with files in subfolder. """
    subfolder = tmp_path / "subfolder"
    subfolder.mkdir()
    image_path = subfolder / "test.jpg"
    image_path.touch()

    result = find_directory_with_files(tmp_path, [".jpg", ".png"])

    # Verify results
    assert result == subfolder, f"Expected {subfolder}, got {result}"


def test_find_directory_with_files_no_files(tmp_path):
    """ Test find_directory_with_files with no matching files. """
    result = find_directory_with_files(tmp_path, [".jpg", ".png"])

    # Verify results
    assert result is None, f"Expected None, got {result}"
#endregion

#region _serialize_obj tests
def test_serialize_obj_primitives():
    """ Test _serialize_obj with primitive types. """
    # Verify results
    assert _serialize_obj(42) == 42
    assert _serialize_obj(3.14) == 3.14
    assert _serialize_obj("test") == "test"
    assert _serialize_obj(True) is True
    assert _serialize_obj(None) is None


def test_serialize_obj_numpy_array():
    """ Test _serialize_obj with NumPy array. """
    array = np.array([1, 2, 3])
    result = _serialize_obj(array)

    # Verify results
    assert result == [1, 2, 3], f"Expected [1, 2, 3], got {result}"


def test_serialize_obj_dict_list():
    """ Test _serialize_obj with dict and list. """
    obj = {"key": [1, 2, {"nested": 3}]}
    result = _serialize_obj(obj)
    expected = {"key": [1, 2, {"nested": 3}]}

    # Verify results
    assert result == expected, f"Expected {expected}, got {result}"


def test_serialize_obj_det_metrics(sample_det_metrics):
    """ Test _serialize_obj with DetMetrics. """
    result = _serialize_obj(sample_det_metrics)

    # Verify results
    assert result["results_dict"]["train/loss"] == 0.5
    assert result["names"] == ["person", "car"]
    assert result["nt_per_class"] == [100, 200]


def test_serialize_obj_unknown_type():
    """ Test _serialize_obj with unknown type. """
    class Unknown:
        pass
    result = _serialize_obj(Unknown())

    # Verify results
    assert isinstance(result, str), f"Expected string, got {type(result)}"
#endregion

#region _compute_iou tests
def test_compute_iou_identical_boxes():
    """ Test _compute_iou with identical boxes. """
    box1 = [0, 0, 100, 100]
    box2 = [0, 0, 100, 100]

    # Verify results
    iou = _compute_iou(box1, box2)
    assert iou == 1.0, f"Expected IoU 1.0 for identical boxes, got {iou}"


def test_compute_iou_no_overlap():
    """ Test _compute_iou with non-overlapping boxes. """
    box1 = [0, 0, 100, 100]
    box2 = [200, 200, 300, 300]

    # Verify results
    iou = _compute_iou(box1, box2)
    assert iou == 0.0, f"Expected IoU 0.0 for non-overlapping boxes, got {iou}"


def test_compute_iou_partial_overlap():
    """ Test _compute_iou with partially overlapping boxes. """
    box1 = [0, 0, 100, 100]
    box2 = [50, 50, 150, 150]

    # Verify results
    iou = _compute_iou(box1, box2)
    expected_iou = 2500 / (10000 + 10000 - 2500) # Intersection: 50x50, Union: 100x100 + 100x100 - 50x50
    assert abs(iou - expected_iou) < 1e-6, f"Expected IoU {expected_iou}, got {iou}"


def test_compute_iou_zero_area():
    """ Test _compute_iou with zero-area box. """
    box1 = [0, 0, 0, 0]
    box2 = [0, 0, 100, 100]

    # Verify results
    iou = _compute_iou(box1, box2)
    assert iou == 0.0, f"Expected IoU 0.0 for zero-area box, got {iou}"
#endregion

#region detection_flicker_rate tests
def test_detection_flicker_rate_empty_detections(empty_detections):
    """ Test detection_flicker_rate with empty detections.  """
    # Verify results
    rate = detection_flicker_rate(empty_detections, iou_thr=0.5)
    assert rate == 0.0, f"Expected flicker rate 0.0 for empty detections, got {rate}"


def test_detection_flicker_rate_single_frame(single_frame_detections):
    """ Test detection_flicker_rate with single frame. """
    # Verify results
    rate = detection_flicker_rate(single_frame_detections, iou_thr=0.5)
    assert rate == 0.0, f"Expected flicker rate 0.0 for single frame, got {rate}"


def test_detection_flicker_rate_no_flicker(sample_detections):
    """ Test detection_flicker_rate with no flicker (high IoU matches). """
    # Verify results
    rate = detection_flicker_rate(sample_detections, iou_thr=0.5)
    expected_flickers = 1 # Only [400, 400, 500, 500] in frame 2 is unmatched
    total_tracks = 5 # 2 (frame 1) + 2 (frame 2) + 1 (frame 3)
    expected_rate = expected_flickers / total_tracks
    assert abs(rate - expected_rate) < 1e-6, f"Expected flicker rate {expected_rate}, got {rate}"


def test_detection_flicker_rate_all_flicker():
    """ Test detection_flicker_rate with all detections unmatched. """
    detections = [
        [[0, 0, 100, 100, 0.9, 0]], # Frame 1
        [[200, 200, 300, 300, 0.8, 1]], # Frame 2
        [[400, 400, 500, 500, 0.7, 1]] # Frame 3
    ]

    # Verify results
    rate = detection_flicker_rate(detections, iou_thr=0.5)
    expected_flickers = 2 # Frame 2 and 3 have no matches with previous frames
    total_tracks = 3 # 1 (frame 1) + 1 (frame 2) + 1 (frame 3)
    expected_rate = expected_flickers / total_tracks
    assert abs(rate - expected_rate) < 1e-6, f"Expected flicker rate {expected_rate}, got {rate}"
#endregion

#region iou_consistency tests
def test_iou_consistency_empty_detections(empty_detections):
    """ Test iou_consistency with empty detections. """
    # Verify results
    consistency = iou_consistency(empty_detections)
    assert consistency == 0.0, f"Expected IoU consistency 0.0 for empty detections, got {consistency}"


def test_iou_consistency_single_frame(single_frame_detections):
    """ Test iou_consistency with single frame. """
    # Verify results
    consistency = iou_consistency(single_frame_detections)
    assert consistency == 0.0, f"Expected IoU consistency 0.0 for single frame, got {consistency}"


def test_iou_consistency_same_class(sample_detections):
    """ Test iou_consistency with same-class detections. """
    consistency = iou_consistency(sample_detections)
    expected_ious = [
        _compute_iou([0, 0, 100, 100], [10, 10, 110, 110]), # Frame 1 -> Frame 2, class 0
        _compute_iou([10, 10, 110, 110], [20, 20, 120, 120]) # Frame 2 -> Frame 3, class 0
    ]
    # Verify results
    expected_mean_iou: float = float(np.mean(expected_ious))
    assert abs(consistency - expected_mean_iou) < 1e-6, f"Expected IoU consistency {expected_mean_iou}, got {consistency}"


def test_iou_consistency_different_classes():
    """ Test iou_consistency with no same-class matches. """
    detections = [
        [[0, 0, 100, 100, 0.9, 0]], # Frame 1, class 0
        [[200, 200, 300, 300, 0.8, 1]] # Frame 2, class 1
    ]

    # Verify results
    consistency = iou_consistency(detections)
    assert consistency == 0.0, f"Expected IoU consistency 0.0 for different classes, got {consistency}"
#endregion
