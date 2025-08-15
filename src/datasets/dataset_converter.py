import re
import shutil
import cv2
import json
import os
from pathlib import Path
from collections import defaultdict
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from PIL import Image

from src.exceptions.exceptions import InvalidInputError
from src.log_config.logging_config import setup_logger

log = setup_logger(__name__)


#region DatasetConverter
class DatasetConverter(ABC):
    """ Abstract class to convert and process dataset to different formats. """

    @abstractmethod
    def convert_annotations(self, dataset_dir: str | Path, cfg: DictConfig) -> None:
        """
        Converts dataset annotations to different model formats.

        :param dataset_dir: Dataset directory
        :param cfg: Hydra configuration
        """
        pass

    @abstractmethod
    def process_dataset(self, dataset_dir: str | Path, cfg: DictConfig) -> None:
        """
        Process datasets.

        :param dataset_dir: Dataset directory
        :param cfg: Hydra configuration
        """
        pass

    def convert_and_process(self, dataset_dir: str | Path, cfg: DictConfig) -> None:
        """
        Convert dataset to different formats.

        :param dataset_dir: Dataset directory
        :param cfg: Hydra configuration
        """
        dataset_dir = Path(dataset_dir)

        log.info(f"Processing {cfg.dataset.name} dataset in {dataset_dir}")
        self.process_dataset(dataset_dir, cfg)
        log.info(f"Finished processing for {cfg.dataset.name} dataset in {dataset_dir}")

        log.info(f"Converting {cfg.dataset.name} dataset in {dataset_dir}")
        self.convert_annotations(dataset_dir, cfg)
        log.info(f"Finished conversion for {cfg.dataset.name} dataset in {dataset_dir}")

#endregion

#region COCOConverter
class COCOConverter(DatasetConverter):
    """ Convert and process COCO dataset to different formats. """

    def process_dataset(self, dataset_dir: str | Path, cfg: DictConfig) -> None:
        """
        Process COCO dataset.

        :param dataset_dir: Dataset directory
        :param cfg: Hydra configuration
        """
        log.info(f"No process methods for COCO dataset")

    def convert_annotations(self, dataset_dir: str | Path, cfg: DictConfig) -> None:
        """ Convert COCO annotations for all models in hydra configuration. """
        dataset_dir = Path(dataset_dir)

        for model_name, model_cfg in cfg.dataset.model.items():
            output_dir = dataset_dir / model_cfg.output_dir
            log.info(f"Converting COCO annotations for model {model_name} in {output_dir}")

            if model_cfg.format == "yolo":
                self._convert_to_yolo(dataset_dir, cfg, output_dir)
            elif model_cfg.format == "coco":
                self._convert_to_coco(dataset_dir, cfg, output_dir)
            else:
                log.error(f"Unknown format {model_cfg.format}")
                raise ValueError(f"Unknown format {model_cfg.format}")

    def _convert_to_yolo(self, dataset_dir: str | Path, cfg: DictConfig, output_dir: Path) -> None:
        """
        Converts COCO annotations to YOLO format.

        :param dataset_dir: Dataset directory
        :param cfg: Hydra configuration
        :param output_dir: Output directory for annotations
        """
        dataset_dir = Path(dataset_dir)

        try:
            if output_dir.exists() and output_dir.is_dir():
                shutil.rmtree(output_dir)
                log.info(f"Removed old YOLO labels folder: {output_dir}")
            else:
                log.info(f"YOLO labels folder does not exist, nothing to remove: {output_dir}")
        except Exception as e:
            log.error(f"Failed to remove old YOLO labels folder {output_dir}: {e}")

        # Process train and val annotations
        for split in ["train", "val"]:
            annotations_path = dataset_dir / cfg.dataset.paths[f"{split}_annotations"]
            images_path = dataset_dir / cfg.dataset.paths[f"{split}_images"]
            output_labels_dir = output_dir / split

            if not annotations_path.exists():
                log.error(f"No annotations found for {split} dataset in {annotations_path}")
                raise InvalidInputError(f"No annotations found for {split} dataset in {annotations_path}")

            os.makedirs(output_labels_dir, exist_ok=True)

            # Load COCO annotations
            with open(annotations_path, "r") as annotations_file:
                coco_data = json.load(annotations_file)

            # Create class mapping based on selected_classes from hydra configuration
            selected_classes = cfg.dataset.classes.selected_classes
            if selected_classes:
                category_map = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
                valid_categories = {k: v for k, v in category_map.items() if v in selected_classes}
                if len(valid_categories) != len(selected_classes):
                    log.error(f"Some selected classes {selected_classes} not found in COCO categories")
                    raise ValueError(f"Some selected classes {selected_classes} not found in COCO categories")
                class_index_map = {cat_id: idx for idx, cat_id in enumerate(valid_categories.keys())}
            else:
                class_index_map = {cat["id"]: idx for idx, cat in enumerate(coco_data["categories"])}

            # Map image IDs to filenames
            image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
            image_annotations = {}
            for ann in coco_data["annotations"]:
                if ann["category_id"] in class_index_map:
                    image_id = ann["image_id"]
                    if image_id not in image_annotations:
                        image_annotations[image_id] = []
                    image_annotations[image_id].append(ann)

            annotation_counter = 0
            for image_id, annotations in image_annotations.items():
                filename = image_id_to_filename.get(image_id)
                if not filename:
                    log.warning(f"Image ID {image_id} not found in COCO images")
                    continue

                img_path = images_path / filename
                if not img_path.exists():
                    log.warning(f"Image {img_path} not found, skipping")
                    continue

                # Get image dimensions
                with Image.open(img_path) as im:
                    img_width, img_height = im.size

                # Convert annotations to YOLO format
                yolo_labels = []
                for ann in annotations:
                    class_id = class_index_map[ann["category_id"]]
                    bbox = ann["bbox"]  # [x_min, y_min, width, height]
                    x_center = (bbox[0] + bbox[2] / 2) / img_width
                    y_center = (bbox[1] + bbox[3] / 2) / img_height
                    width = bbox[2] / img_width
                    height = bbox[3] / img_height
                    yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                if yolo_labels:
                    annotation_counter += 1
                    label_filename = Path(filename).stem + ".txt"
                    output_path = output_labels_dir / label_filename
                    with open(output_path, "w") as f:
                        f.write("\n".join(yolo_labels))

            log.info(f"Generated {annotation_counter} label files for {split} in {output_labels_dir}")

    def _convert_to_coco(self, dataset_dir: str | Path, cfg: DictConfig, output_dir: Path) -> None:
        """
        Converts COCO annotations to COCO format for Faster-RCNN.

        :param dataset_dir: Dataset directory
        :param cfg: Hydra configuration
        :param output_dir: Output directory for annotations
        """
        dataset_dir = Path(dataset_dir)

        for split in ["train", "val"]:
            output_annotations_path = output_dir / f"frcnn_instances_{split}2017.json"
            try:
                if output_annotations_path.exists():
                    output_annotations_path.unlink()
                    log.info(f"Removed old Faster-RCNN annotations file: {output_annotations_path}")
                else:
                    log.info(
                        f"Faster-RCNN annotations file does not exist, nothing to remove: {output_annotations_path}")
            except Exception as e:
                log.error(f"Failed to remove {output_annotations_path}: {e}")

        # Process train and val annotations
        for split in ["train", "val"]:
            annotations_path = dataset_dir / cfg.dataset.paths[f"{split}_annotations"]
            output_annotations_path = output_dir / f"frcnn_instances_{split}2017.json"

            if not annotations_path.exists():
                log.error(f"Annotations file {annotations_path} not found")
                raise InvalidInputError(f"Annotations file {annotations_path} not found")

            # Load COCO annotations
            with open(annotations_path, "r") as f:
                coco_data = json.load(f)

            # Create class mapping based on selected_classes from hydra configuration
            selected_classes = cfg.dataset.classes.selected_classes
            if selected_classes:
                category_map = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
                valid_categories = {k: v for k, v in category_map.items() if v in selected_classes}
                if len(valid_categories) != len(selected_classes):
                    log.error(f"Some selected classes {selected_classes} not found in COCO categories")
                    raise ValueError(f"Some selected classes {selected_classes} not found in COCO categories")
                id_mapping = {cat_id: idx + 1 for idx, cat_id in enumerate(valid_categories.keys())}
            else:
                id_mapping = {cat["id"]: cat["id"] for cat in coco_data["categories"]}

            # Filter categories
            categories = [
                {"id": id_mapping[cat["id"]], "name": cat["name"], "supercategory": cat.get("supercategory", "")}
                for cat in coco_data["categories"] if cat["id"] in id_mapping
            ]

            # Filter annotations
            annotations = []
            for ann in coco_data["annotations"]:
                if ann["category_id"] in id_mapping:
                    new_ann = ann.copy()
                    new_ann["category_id"] = id_mapping[ann["category_id"]]
                    # Skip invalid annotations
                    if ann["bbox"][3] > 0:
                        annotations.append(new_ann)

            # Filter images to include only those with valid annotations
            image_ids = set(ann["image_id"] for ann in annotations)
            images = [img for img in coco_data["images"] if img["id"] in image_ids]

            # Create and save new COCO JSON
            new_coco = {
                "images": images,
                "annotations": annotations,
                "categories": categories
            }

            os.makedirs(output_dir, exist_ok=True)
            with open(output_annotations_path, "w") as f:
                json.dump(new_coco, f)
            log.info(f"Saved filtered COCO annotations for {split} to {output_annotations_path}")

#endregion

#region BDD100KConverter
class BDD100KConverter(DatasetConverter):
    # Classes mapping
    _BDD_TO_COCO = {
        "pedestrian": "person",
        "rider": "person",
        "other person": "person",
        "bicycle": "bicycle",
        "car": "car",
        "motorcycle": "motorcycle",
        "bus": "bus",
        "train": "train",
        "truck": "truck"
    }

    _COCO_CATEGORIES = [
        {"id": 1, "name": "person"},
        {"id": 2, "name": "bicycle"},
        {"id": 3, "name": "car"},
        {"id": 4, "name": "motorcycle"},
        {"id": 5, "name": "bus"},
        {"id": 6, "name": "train"},
        {"id": 7, "name": "truck"},
    ]

    _COCO_NAME_TO_ID = {cat["name"]: cat["id"] for cat in _COCO_CATEGORIES}
    orig_width, orig_height = 1280, 720

    def process_dataset(self, dataset_dir: str | Path, cfg: DictConfig) -> None:
        """
        Process BDD100K dataset.

        :param dataset_dir: Dataset directory
        :param cfg: Hydra configuration
        """
        dataset_dir = Path(dataset_dir)
        target_size = tuple[int, int](cfg.dataset.image_target_size)
        fps = cfg.dataset.video_fps

        # Process train and val splits
        for split in ["train", "val"]:
            images_path = dataset_dir / cfg.dataset.paths[f"{split}_images"]
            videos_path = dataset_dir / cfg.dataset.paths[f"{split}_videos"]

            if not images_path.exists():
                log.error(f"Images directory {images_path} does not exist")
                raise InvalidInputError(f"Images directory {images_path} does not exist")

            # Resize images
            log.info(f"Resizing images for {split} in {images_path}")
            self._resize_images_in_folders(images_path, split, target_size)

            # Create videos
            if split == "val":
                log.info(f"Creating videos for val in {videos_path}")
                self._create_videos(images_path, videos_path, fps)

    def convert_annotations(self, dataset_dir: str | Path, cfg: DictConfig) -> None:
        """
        Convert BDD100K annotations for all models in hydra configuration.

        :param dataset_dir: Dataset directory
        :param cfg: Hydra configuration
        """
        dataset_dir = Path(dataset_dir)

        # Validate selected classes
        selected_classes = cfg.dataset.classes.selected_classes
        if selected_classes:
            invalid_classes = [cls for cls in selected_classes if cls not in self._BDD_TO_COCO.values()]
            if invalid_classes:
                log.error(f"Invalid selected classes: {invalid_classes}")
                raise ValueError(f"Invalid selected classes: {invalid_classes}")

        for model_name, model_cfg in cfg.dataset.model.items():
            output_dir = dataset_dir / model_cfg.output_dir
            log.info(f"Converting BDD100K annotations for {model_name} in {output_dir}")

            if model_cfg.format == "yolo":
                self._convert_to_yolo(dataset_dir, cfg, output_dir)
            elif model_cfg.format == "coco":
                self._convert_to_coco(dataset_dir, cfg, output_dir)
            else:
                log.error(f"Unknown format {model_cfg.format}")
                raise ValueError(f"Unknown format {model_cfg.format}")

    def _process_frame(self, frame: dict, image_dir: Path, target_width: int, target_height: int,
                      selected_classes: list, format_type: str, video_name: str) -> tuple:
        """
        Process a single frame and return annotations in specified format.

        :param frame: Frame data from JSON
        :param image_dir: Directory with images
        :param target_width: Target image width
        :param target_height: Target image height
        :param selected_classes: List of selected classes
        :param format_type: Output format ('coco' or 'yolo')
        :param video_name: Name of the video
        :return: Tuple of image_data, annotations, empty_frame, skipped_boxes, skipped_categories
        """
        image_name = frame.get("name")
        if not image_name:
            log.warning(f"Skipping frame with missing name")
            return None, [], False, 0, set()

        if not video_name:
            log.warning(f"Skipping frame with missing videoName")
            return None, [], False, 0, set()

        file_name = f"{video_name}/{image_name}"
        image_path = image_dir / file_name
        if not image_path.exists():
            log.warning(f"Image {image_path} does not exist")
            return None, [], False, 0, set()

        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            log.warning(f"Failed to read image {image_path}: {e}")
            return None, [], False, 0, set()

        if width != target_width or height != target_height:
            log.warning(f"Image {image_path} has unexpected size: {width}x{height}, expected {target_width}x{target_height}")
            return None, [], False, 0, set()

        image_data = {"file_name": file_name, "width": width, "height": height} if format_type == "coco" else {"txt_path": image_path.with_suffix(".txt")}

        labels = frame.get("labels")
        if not labels:
            return image_data, [], True, 0, set()

        annotations = []
        skipped_boxes = 0
        skipped_categories = set()

        for label in labels:
            if not isinstance(label, dict):
                log.warning(f"Invalid label format for {image_name}: expected dict, got {type(label)}")
                continue
            category = label.get("category")
            if not category or category not in self._BDD_TO_COCO:
                skipped_categories.add(category)
                continue
            mapped_category = self._BDD_TO_COCO[category]
            if mapped_category not in selected_classes:
                skipped_categories.add(category)
                continue
            class_id = self._COCO_NAME_TO_ID[mapped_category]
            box = label.get("box2d")
            if not box:
                skipped_boxes += 1
                continue
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            width_bbox = x2 - x1
            height_bbox = y2 - y1
            if width_bbox <= 0 or height_bbox <= 0:
                skipped_boxes += 1
                continue

            x1_new = x1 * target_width / self.orig_width
            y1_new = y1 * target_height / self.orig_height
            x2_new = x2 * target_width / self.orig_width
            y2_new = y2 * target_height / self.orig_height
            width_bbox_new = x2_new - x1_new
            height_bbox_new = y2_new - y1_new

            if format_type == "coco":
                annotations.append({
                    "category_id": class_id,
                    "bbox": [x1_new, y1_new, width_bbox_new, height_bbox_new],
                    "area": width_bbox_new * height_bbox_new,
                    "iscrowd": 0
                })
            elif format_type == "yolo":
                x_center = (x1_new + width_bbox_new / 2) / target_width
                y_center = (y1_new + height_bbox_new / 2) / target_height
                width_norm = width_bbox_new / target_width
                height_norm = height_bbox_new / target_height

                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width_norm <= 1 and 0 <= height_norm <= 1):
                    log.warning(f"Out of bounds in {image_data['txt_path']}: x_center={x_center:.4f}, y_center={y_center:.4f}, width={width_norm:.4f}, height={height_norm:.4f}")
                    skipped_boxes += 1
                    continue

                annotations.append(f"{class_id - 1} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")
            else:
                log.error(f"Invalid format: {format_type}")
                raise InvalidInputError(f"Invalid format: {format_type}")

        return image_data, annotations, False, skipped_boxes, skipped_categories

    def _convert_to_coco(self, dataset_dir: Path, cfg: DictConfig, output_dir: Path) -> None:
        """
        Convert BDD100K annotations to COCO format, scaling from original to target size.

        :param dataset_dir: Dataset directory
        :param cfg: Hydra configuration
        :param output_dir: Output directory for annotations
        """
        target_width, target_height = cfg.dataset.image_target_size
        selected_classes = cfg.dataset.classes.selected_classes

        for split in ["train", "val"]:
            json_dir = dataset_dir / cfg.dataset.paths[f"{split}_annotations"]
            image_dir = dataset_dir / cfg.dataset.paths[f"{split}_images"]
            output_file = output_dir / f"{split}.json"

            if not json_dir.exists():
                log.error(f"Annotations directory {json_dir} does not exist")
                raise InvalidInputError(f"Annotations directory {json_dir} does not exist")
            if not image_dir.exists():
                log.error(f"Images directory {image_dir} does not exist")
                raise InvalidInputError(f"Images directory {image_dir} does not exist")

            os.makedirs(output_dir, exist_ok=True)
            json_files = sorted(json_dir.glob("*.json"))
            log.info(f"[{split.upper()}] Processing {len(json_files)} JSON files")

            coco_data = {
                "images": [],
                "annotations": [],
                "categories": self._COCO_CATEGORIES
            }
            image_id = 0
            annotation_id = 0
            total_annotations = 0
            empty_frames = 0
            skipped_boxes = 0
            skipped_categories = set()

            for json_path in json_files:
                video_name = json_path.stem
                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)
                except Exception as e:
                    log.error(f"Failed to read JSON {json_path}: {e}")
                    continue

                if not isinstance(data, list):
                    log.error(f"Invalid JSON structure in {json_path}: expected list, got {type(data)}")
                    continue

                frame_count = 0
                for frame in data:
                    if not isinstance(frame, dict):
                        log.error(f"Invalid frame in {json_path}: expected dict, got {type(frame)}")
                        continue

                    image_data, annotations, is_empty, frame_skipped_boxes, frame_skipped_categories = self._process_frame(
                        frame, image_dir, target_width, target_height, selected_classes, "coco", video_name
                    )
                    if image_data is None:
                        continue

                    frame_count += 1
                    if is_empty:
                        empty_frames += 1
                    skipped_boxes += frame_skipped_boxes
                    skipped_categories.update(frame_skipped_categories)

                    coco_data["images"].append({
                        "id": image_id,
                        "file_name": image_data["file_name"],
                        "width": image_data["width"],
                        "height": image_data["height"]
                    })
                    for ann in annotations:
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            **ann
                        })
                        annotation_id += 1
                        total_annotations += 1
                    image_id += 1

            log.info(
                f"[{split.upper()}] Created {len(coco_data['images'])} images, "
                f"{total_annotations} annotations, {skipped_boxes} skipped bboxes, "
                f"{empty_frames} empty frames"
            )
            if skipped_categories:
                log.info(f"[{split.upper()}] Skipped categories: {sorted(skipped_categories)}")
            with open(output_file, "w") as f:
                json.dump(coco_data, f, indent=2)

    def _convert_to_yolo(self, dataset_dir: Path, cfg: DictConfig, output_dir: Path) -> None:
        """
        Convert BDD100K annotations to YOLO format, scaling from original to target size.

        :param dataset_dir: Dataset directory
        :param cfg: Hydra configuration
        :param output_dir: Output directory for annotations
        """
        target_width, target_height = cfg.dataset.image_target_size
        selected_classes = cfg.dataset.classes.selected_classes

        for split in ["train", "val"]:
            json_dir = dataset_dir / cfg.dataset.paths[f"{split}_annotations"]
            image_dir = dataset_dir / cfg.dataset.paths[f"{split}_images"]
            txt_output_dir = output_dir / split

            if not json_dir.exists():
                log.error(f"Annotations directory {json_dir} does not exist")
                raise InvalidInputError(f"Annotations directory {json_dir} does not exist")
            if not image_dir.exists():
                log.error(f"Images directory {image_dir} does not exist")
                raise InvalidInputError(f"Images directory {image_dir} does not exist")

            os.makedirs(txt_output_dir, exist_ok=True)
            json_files = sorted(json_dir.glob("*.json"))
            log.info(f"[{split.upper()}] Processing {len(json_files)} JSON files")

            total_annotations = 0
            empty_frames = 0
            skipped_boxes = 0
            skipped_categories = set()

            for json_path in json_files:
                video_name = json_path.stem
                target_folder = txt_output_dir / video_name
                target_folder.mkdir(parents=True, exist_ok=True)

                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)
                except Exception as e:
                    log.error(f"Failed to read JSON {json_path}: {e}")
                    continue

                if not isinstance(data, list):
                    log.error(f"Invalid JSON structure in {json_path}: expected list, got {type(data)}")
                    continue

                frame_count = 0
                for frame in data:
                    if not isinstance(frame, dict):
                        log.error(f"Invalid frame in {json_path}: expected dict, got {type(frame)}")
                        continue

                    image_name = frame.get("name")
                    if not image_name:
                        log.warning(f"Skipping frame with missing name in {json_path}")
                        continue
                    if not video_name:
                        log.warning(f"Skipping frame with missing videoName in {json_path}")
                        continue
                    txt_path = target_folder / Path(image_name).with_suffix(".txt")

                    image_data, annotations, is_empty, frame_skipped_boxes, frame_skipped_categories = self._process_frame(
                        frame, image_dir, target_width, target_height, selected_classes, "yolo", video_name
                    )
                    if image_data is None:
                        continue

                    frame_count += 1
                    if is_empty:
                        empty_frames += 1
                        txt_path.touch()
                        log.debug(f"Created empty annotation file: {txt_path}")
                    else:
                        with open(txt_path, "w") as f:
                            f.write("\n".join(annotations))
                        total_annotations += len(annotations)
                        log.debug(f"Wrote {len(annotations)} annotations to {txt_path}")
                    skipped_boxes += frame_skipped_boxes
                    skipped_categories.update(frame_skipped_categories)

            log.info(
                f"[{split.upper()}] Created {total_annotations} annotations, "
                f"{empty_frames} empty frames, {skipped_boxes} skipped bboxes"
            )
            if skipped_categories:
                log.info(f"[{split.upper()}] Skipped categories: {sorted(skipped_categories)}")

    def _resize_images_in_folders(self, images_path: Path, split_name: str, target_size: tuple[int, int]) -> None:
        """
        Resize all images in subfolders to target size.

        :param images_path: Path to images directory
        :param split_name: Dataset split (train/val)
        :param target_size: Target size for resizing (width, height)
        """
        folder_count = 0
        for subfolder in images_path.iterdir():
            if subfolder.is_dir():
                folder_count += 1
                for image_path in subfolder.glob("*.jpg"):
                    try:
                        with Image.open(image_path) as img:
                            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                            img_resized.save(image_path, quality=95)
                    except Exception as e:
                        log.warning(f"Failed to process image {image_path}: {e}")
                log.info(f"[{split_name.upper()}] ({folder_count}) Finished processing folder: {subfolder.name}")

    def _create_videos(self, images_path: Path, videos_path: Path, fps: int) -> None:
        """
        Create videos from images in subfolders.

        :param images_path: Path to images directory
        :param videos_path: Path to output videos directory
        :param fps: Frames per second for videos
        """
        os.makedirs(videos_path, exist_ok=True)
        video_frames = defaultdict(list)

        for frame_path in images_path.glob("**/*.jpg"):
            video_name = frame_path.parent.name
            video_frames[video_name].append(str(frame_path))

        for video_name, frame_paths in video_frames.items():
            if not frame_paths:
                log.warning(f"No frames found for video: {video_name}")
                continue

            frame_paths.sort(
                key=lambda x: int(re.search(r'(\d+)\.jpg$', x).group(1)) if re.search(r'(\d+)\.jpg$', x) else 0)

            first_frame = cv2.imread(frame_paths[0])
            if first_frame is None:
                log.error(f"Could not read first frame: {frame_paths[0]}, skipping {video_name}")
                continue
            height, width = first_frame.shape[:2]

            output_video_path = videos_path / f"{video_name}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                if frame is None:
                    log.warning(f"Skipping invalid frame: {frame_path}")
                    continue
                video_writer.write(frame)

            video_writer.release()
            log.info(f"Created video: {output_video_path}")

#endregion
