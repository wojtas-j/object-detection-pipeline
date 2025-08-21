import pytest
import numpy as np

from src.log_config.train_eval_metric_utils import detection_flicker_rate, iou_consistency, _compute_iou

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

#endregion

#region Compute IoU tests
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

#region DFR tests
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

#region IoU Consistency tests
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
