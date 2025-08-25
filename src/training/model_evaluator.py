import time
import cv2
import numpy as np
import torch
from pathlib import Path
from abc import ABC, abstractmethod
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

from src.log_config.train_eval_metric_utils import detection_flicker_rate, iou_consistency, save_yolo_metrics, \
    predict_yolo_evaluation_image, predict_yolo_evaluation_video, find_directory_with_files
from src.datasets.datasets_utils import path_exists
from src.exceptions.exceptions import EvaluationError, YamlConfigError, MetricsLoggingError
from src.log_config.logging_config import setup_logger

log = setup_logger(name=__name__)


#region ModelEvaluator
class ModelEvaluator(ABC):
    """ Abstract class for evaluating different models. """

    @abstractmethod
    def evaluate_model(self, cfg: DictConfig) -> None:
        """
        Evaluate a model using specified configuration file on specified dataset.

        :param cfg: Hydra configuration file.
        """
        pass

    @abstractmethod
    def log_final_metrics(self, cfg: DictConfig, results: DetMetrics | dict) -> None:
        """
        Log final metrics after evaluation.

        :param cfg: Hydra configuration file.
        :param results: Evaluation metrics results.
        """
        pass

#endregion

#region YOLOEvaluator
class YOLOEvaluator(ModelEvaluator):
    """ Evaluator class for running YOLO model on specified dataset. """
    def evaluate_model(self, cfg: DictConfig) -> None:
        """
        Evaluate a model using specified configuration file on specified dataset.

        :param cfg: Hydra configuration file.
        """
        if cfg.evaluation.device == 0 and not torch.cuda.is_available():
            log.error("CUDA is not available, cannot evaluate model")
            raise EvaluationError("CUDA is not available")

        model_path = cfg.evaluation.model_path
        if model_path is None:
            log.error(f"Invalid or missing model_path: {model_path}")
            raise YamlConfigError(f"Invalid or missing model_path: {model_path}")
        if not path_exists(model_path):
            log.error(f"Model file not found: {model_path}")
            raise YamlConfigError(f"Model file not found: {model_path}")

        videos_path = cfg.evaluation.videos_path
        if cfg.evaluation.eval_videos:
            if videos_path is None:
                log.error(f"Invalid or missing videos_path: {videos_path}")
                raise YamlConfigError(f"Invalid or missing videos_path: {videos_path}")
            if not path_exists(videos_path):
                log.error(f"Videos file not found: {videos_path}")
                raise YamlConfigError(f"Videos file not found: {videos_path}")
            video_files = list(Path(cfg.evaluation.videos_path).glob("*.mp4"))
            if not video_files:
                log.error(f"No videos found in {cfg.evaluation.videos_path}")
                raise YamlConfigError(f"No videos found in {cfg.evaluation.videos_path}")

        # Evaluation
        model = YOLO(model_path)
        # Evaluation on images
        eval_params = self._get_eval_params(cfg)
        start_time = time.time()
        try:
            log.info(f"Started evaluation on images")
            results_image = model.val(**eval_params)
        except (EvaluationError, YamlConfigError, RuntimeError) as e:
            end_time = time.time()
            total_time = end_time - start_time
            log.error(f"Evaluation failed after {total_time:.2f}: {e}")
            raise
        else:
            end_time = time.time()
            total_time = end_time - start_time
            log.info(f"Evaluation {model_path} on images finished {cfg.evaluation.name} in {total_time:.2f} seconds.")
            self.log_final_metrics(cfg, results_image)

        # Evaluation on videos
        if cfg.evaluation.eval_videos:
            eval_params = self._get_eval_params(cfg)
            start_time = time.time()
            try:
                log.info(f"Started evaluation on videos")
                results_videos = self._evaluate_on_videos(model, cfg, eval_params)
            except (EvaluationError, YamlConfigError, RuntimeError) as e:
                end_time = time.time()
                total_time = end_time - start_time
                log.error(f"Evaluation failed after {total_time:.2f}: {e}")
                raise
            else:
                end_time = time.time()
                total_time = end_time - start_time
                log.info(f"Evaluation {model_path} on videos finished {cfg.evaluation.name} in {total_time:.2f} seconds.")
                self.log_final_metrics(cfg, results_videos)

    def log_final_metrics(self, cfg: DictConfig, results: DetMetrics | dict) -> None:
        """
        Log final metrics after evaluation.

        :param cfg: Hydra configuration file.
        :param results: Evaluation metrics results.
        """
        if isinstance(results, DetMetrics): # YOLO (images)
            log.info(f"Logging final metrics after evaluation {cfg.evaluation.name} for images")

            # Save metrics results
            path = Path(cfg.evaluation.project) / cfg.evaluation.name
            save_yolo_metrics(results, path, "yolo_image_evaluation_results.json")

            # Save prediction for image
            data_cfg = OmegaConf.load(cfg.evaluation.data)
            path = find_directory_with_files(Path("datasets") / data_cfg.path / data_cfg.val, [".jpg", ".png"])
            if path is None:
                log.error(f"Invalid path to predict image.")
                return
            else:
                predict_yolo_evaluation_image(cfg, path)
        elif isinstance(results, dict): # YOLO (videos)
            log.info(f"Logging final metrics after evaluation {cfg.evaluation.name} for videos")

            # Save metrics results
            path = Path(cfg.evaluation.project) / cfg.evaluation.name
            save_yolo_metrics(results, path, "yolo_videos_evaluation_results.json")

            # Save prediction for video
            path = find_directory_with_files(Path(cfg.evaluation.videos_path), [".mp4"])
            if path is None:
                log.error(f"Invalid path to predict video.")
                return
            else:
                predict_yolo_evaluation_video(cfg, path)
        else:
            log.error(f"Invalid type of final metrics results: {results}, type: {type(results)}")
            raise MetricsLoggingError(f"Invalid type of final metrics results: {results}, type: {type(results)}")

    def _get_eval_params(self, cfg: DictConfig) -> dict:
        """
        Prepare evaluation parameters.

        :param cfg: Hydra configuration file.
        :return: Dictionary of evaluation parameters.
        """
        return {
            "name": cfg.evaluation.name,
            "project": cfg.evaluation.project,
            "data": cfg.evaluation.data,
            "imgsz": cfg.evaluation.imgsz,
            "batch": cfg.evaluation.batch,
            "device": cfg.evaluation.device,
            "workers": cfg.evaluation.workers,
            "conf": cfg.evaluation.conf,
            "iou": cfg.evaluation.iou
        }

    def _evaluate_on_videos(self, model: YOLO, cfg: DictConfig, eval_parameters: dict) -> dict:
        """
        Evaluate a YOLO model on a set of video files and compute aggregated metrics.

        :param model: Trained YOLO model.
        :param cfg: Hydra configuration file.
        :param eval_parameters: Additional parameters for evaluation
        :return: Dictionary of evaluation results.
        """
        video_path = cfg.evaluation.videos_path
        video_dir = Path(video_path)
        video_files = list(video_dir.glob("*.mp4"))

        all_times = []
        all_flickers = []
        all_ious = []

        for video_path in video_files:
            cap = cv2.VideoCapture(str(video_path))
            frame_times = []
            frame_detections = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (eval_parameters["imgsz"], eval_parameters["imgsz"]))
                img = img / 255.0
                img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
                img = img.to(eval_parameters["device"])

                start = time.time()
                preds = model(img, conf=eval_parameters["conf"], iou=eval_parameters["iou"], verbose=False)
                end = time.time()
                frame_times.append((end - start) * 1000)

                dets = []
                for r in preds[0].boxes:
                    x1, y1, x2, y2 = r.xyxy[0].tolist()
                    conf_ = r.conf[0].item()
                    cls_ = r.cls[0].item()
                    dets.append([x1, y1, x2, y2, conf_, cls_])
                frame_detections.append(dets)

            cap.release()
            flicker = detection_flicker_rate(frame_detections, eval_parameters["iou"])
            all_flickers.append(flicker)
            iou_cons = iou_consistency(frame_detections)
            all_ious.append(iou_cons)
            all_times.extend(frame_times)

        avg_time = np.mean(all_times) if all_times else 0.0
        avg_flicker = np.mean(all_flickers) if all_flickers else 0.0
        avg_iou_cons = np.mean(all_ious) if all_ious else 0.0
        log.info(f"Average frame processing time: {avg_time:.2f} ms")
        log.info(f"Average detection flicker rate: {avg_flicker:.4f}")
        log.info(f"Average IoU consistency: {avg_iou_cons:.4f}")

        return {
            "average_time_ms": avg_time,
            "average_flicker_rate": avg_flicker,
            "average_iou_consistency": avg_iou_cons,
            "num_videos": len(video_files),
            "frames_evaluated": len(all_times)
        }

#endregion
