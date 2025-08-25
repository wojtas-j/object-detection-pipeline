import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Any
from omegaconf import DictConfig
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics, Metric

from src.exceptions.exceptions import YamlConfigError, PredictionError
from src.log_config.logging_config import setup_logger

log = setup_logger(name=__name__)


def save_yolo_metrics(results: DetMetrics | dict, path_to_save: str | Path, file_name: str) -> None:
    """
    Save all YOLO training metrics to a JSON file.

    :param results: YOLO training metrics results.
    :param path_to_save: Path to save the JSON file.
    :param file_name: JSON file name to save.
    """
    path = Path(path_to_save) / file_name
    serializable = _serialize_obj(results)

    # Save to JSON
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=4)

    log.info(f"YOLO metrics {file_name} saved to {path}")


def predict_yolo_evaluation_image(cfg: DictConfig, images_path: str | Path) -> None:
    """
    Save YOLO prediction for one image from dataset.

    :param cfg: Hydra configuration.
    :param images_path: Path to images folder.
    """
    images_files = sorted(images_path.glob("*.[jp][pn]g"))
    model = YOLO(cfg.evaluation.model_path)
    image_to_predict_path = images_files[0]
    predict_directory_to = Path(cfg.evaluation.project) / cfg.evaluation.name
    prediction_name = f"{cfg.evaluation.name}_predict"
    try:
        results = model.predict(
            name = prediction_name,
            project=str(predict_directory_to),
            source=str(image_to_predict_path),
            imgsz=cfg.evaluation.imgsz,
            conf=cfg.evaluation.conf,
            save=False,
            verbose=False,
            exist_ok=True
        )
        predicted_image = results[0].plot()
        output_folder = Path(predict_directory_to) / prediction_name
        output_folder.mkdir(parents=True, exist_ok=True)
        path_to_save =  output_folder / f"{Path(results[0].path).stem}_predict.jpg"
        cv2.imwrite(str(path_to_save), predicted_image)
        log.info(f"Prediction image {prediction_name} saved to {path_to_save}")
    except (PredictionError, YamlConfigError, Exception, RuntimeError) as e:
        log.error(f"Image prediction error: {e}")
        return


def predict_yolo_evaluation_video(cfg: DictConfig, videos_path: str | Path) -> None:
    """
    Save YOLO prediction for one video from dataset.

    :param cfg: Hydra configuration.
    :param videos_path: Path to videos folder.
    """
    model = YOLO(cfg.evaluation.model_path)
    videos_files = sorted(videos_path.glob("*.mp4"))
    video_to_predict_path = videos_files[0]
    predict_directory_to = Path(cfg.evaluation.project) / cfg.evaluation.name
    prediction_name = f"{cfg.evaluation.name}_predict"
    output_folder = Path(predict_directory_to) / prediction_name
    output_folder.mkdir(parents=True, exist_ok=True)
    try:
        results = model.predict(
            name=prediction_name,
            project=str(predict_directory_to),
            source=str(video_to_predict_path),
            imgsz=cfg.evaluation.imgsz,
            conf=cfg.evaluation.conf,
            save=True,
            verbose=False,
            exist_ok=True
        )

        log.info(f"Prediction video {prediction_name} saved to {results[0].path}")
    except (PredictionError, YamlConfigError, Exception, RuntimeError) as e:
        log.error(f"Video prediction error: {e}")
        return


def find_directory_with_files(base_folder: str | Path, extensions: List[str]) -> Path | None:
    """
    Search base_folder and subfolders for base_folder recursively and find any files with the given extensions.

    :param base_folder: Base folder path.
    :param extensions: Extensions to search for.
    :return: Path to the folder containing files with the given extensions.
    """
    base_folder = Path(base_folder)
    # Check in the base_folder
    for extension in extensions:
        if any(base_folder.glob(f"*{extension}")):
            return base_folder

    # Check in the base_folder subfolders
    for subfolder in sorted(f for f in base_folder.iterdir() if f.is_dir()):
        for extension in extensions:
            if any(subfolder.glob(f"*{extension}")):
                return subfolder

    log.error(f"No files with the given extensions: {extensions} found in {base_folder} and subfolders")
    return None


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


def _serialize_obj(obj: Any) -> dict | list | str | int | float | bool | None:
    """
    Recursively converts objects (including Ultralytics `DetMetrics` and `Metric`) into JSON-serializable structures such as dicts, lists, and primitives.

    :param obj: Object to serialize.
    :return: A JSON-serializable representation of the input object, which may be a dict, list, primitive, or string fallback.
    """
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _serialize_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_obj(v) for v in obj]
    if isinstance(obj, DetMetrics):
        return {
            "results_dict": _serialize_obj(obj.results_dict),
            "curves_results": _serialize_obj(obj.curves_results),
            "names": _serialize_obj(obj.names),
            "nt_per_class": _serialize_obj(obj.nt_per_class),
            "ap_class_index": _serialize_obj(obj.ap_class_index),
        }
    if isinstance(obj, Metric):
        return _serialize_obj(obj.__dict__)

    return str(obj)
#endregion