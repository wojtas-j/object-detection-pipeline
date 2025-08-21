import numpy as np
from typing import List

from src.log_config.logging_config import setup_logger

log = setup_logger(name=__name__)


def detection_flicker_rate(detections_per_frame: List[List[List[float]]], iou_thr: float) -> float:
    """
    Compute the detection flicker rate over frames.
    Flicker: detections in the current frame that do not match any detection in the previous frame.

    :param detections_per_frame: list of detections per frame, each detection is [x1, y1, x2, y2, conf, cls].
    :param iou_thr: IOU threshold to consider a detection as matched.
    :return: flicker rate (float).
    """
    if len(detections_per_frame) < 2:
        return 0.0

    flickers = 0
    total_tracks = 0
    prev_boxes = []
    for frame_idx, dets in enumerate(detections_per_frame):
        curr_boxes = [d[:4] for d in dets]  # extract bounding boxes only

        total_tracks += len(curr_boxes)

        if frame_idx == 0:
            prev_boxes = curr_boxes
            continue

        for box in curr_boxes:
            matched = False
            for pbox in prev_boxes:
                iou = _compute_iou(box, pbox)
                if iou > iou_thr:
                    matched = True
                    break
            if not matched:
                flickers += 1

        prev_boxes = curr_boxes

    flicker_rate = flickers / total_tracks if total_tracks > 0 else 0.0

    return flicker_rate


def iou_consistency(detections_per_frame: List[List[List[float]]]) -> float:
    """
    Compute mean IoU consistency between consecutive frames for objects of the same class.

    :param detections_per_frame: list of detections per frame, each detection is [x1, y1, x2, y2, conf, cls].
    :return: mean IoU consistency (float).
    """
    if len(detections_per_frame) < 2:
        return 0.0

    ious = []

    for t in range(len(detections_per_frame) - 1):
        dets_t = detections_per_frame[t]
        dets_tp1 = detections_per_frame[t + 1]

        for det in dets_t:
            box1, cls1 = det[:4], det[5]
            best_iou = 0.0
            for det2 in dets_tp1:
                box2, cls2 = det2[:4], det2[5]
                if cls1 == cls2:  # only compare same-class objects
                    iou = _compute_iou(box1, box2)
                    if iou > best_iou:
                        best_iou = iou
            if best_iou > 0:
                ious.append(best_iou)

    mean_iou = np.mean(ious) if ious else 0.0

    return mean_iou


def _compute_iou(box1: list[float], box2: list[float]) -> float:
    """
    Compute the IoU between two bounding boxes.

    :param box1: Bounding box in the format [x1, y1, x2, y2].
    :param box2: Bounding box in the format [x1, y1, x2, y2].
    :return: IoU value between the two boxes, in the range [0.0, 1.0].
    """
    x_a = max(box1[0], box2[0])
    y_a = max(box1[1], box2[1])
    x_b = min(box1[2], box2[2])
    y_b = min(box1[3], box2[3])
    inter_area = max(0.0, float(x_b - x_a)) * max(0.0, float(y_b - y_a))
    box_a_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box_b_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    denom = box_a_area + box_b_area - inter_area
    iou = inter_area / denom if denom > 0 else 0.0
    
    return iou