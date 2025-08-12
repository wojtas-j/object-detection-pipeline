import pytest
import json
from pathlib import Path
from unittest.mock import patch, Mock, mock_open
from PIL import Image
from omegaconf import DictConfig

from src.datasets.dataset_converter import DatasetConverter, COCOConverter, BDD100KConverter
from src.exceptions.exceptions import InvalidInputError

#region Fixtures
@pytest.fixture
def temp_dir(tmp_path):
    """ Create a temporary directory. """
    return tmp_path


@pytest.fixture
def mock_coco_hydra_config():
    """ Mock Hydra configuration for COCO"""
    return DictConfig({
        "dataset": {
            "name": "coco",
            "classes": {
                "selected_classes": ["person", "car"]
            },
            "model": {
                "yolo": {
                    "format": "yolo",
                    "output_dir": "yolo"
                },
                "faster_rcnn": {
                    "format": "coco",
                    "output_dir": "faster_rcnn"
                }
            },
            "paths": {
                "train_images": "train2017",
                "val_images": "val2017",
                "train_annotations": "annotations/instances_train2017.json",
                "val_annotations": "annotations/instances_val2017.json"
            }
        }
    })


@pytest.fixture
def mock_coco_annotations():
    """ Mock COCO annotations. """
    return {
        "images": [
            {"id": 1, "file_name": "000000000001.jpg", "width": 640, "height": 640},
            {"id": 2, "file_name": "000000000002.jpg", "width": 640, "height": 640}
        ],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50], "id": 1},
            {"image_id": 1, "category_id": 2, "bbox": [200, 200, 100, 100], "id": 2}
        ],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "car"}
        ]
    }


@pytest.fixture
def mock_bdd100k_hydra_config():
    """ Mock Hydra configuration for BDD100K. """
    return DictConfig({
        "dataset": {
            "name": "bdd100k",
            "image_target_size": [640, 640],
            "video_fps": 30,
            "classes": {
                "selected_classes": ["person", "car"]
            },
            "paths": {
                "train_images": "images/train",
                "val_images": "images/val",
                "train_annotations": "annotations/train",
                "val_annotations": "annotations/val",
                "train_videos": "videos/train",
                "val_videos": "videos/val"
            },
            "model": {
                "yolo": {
                    "format": "yolo",
                    "output_dir": "yolo"
                },
                "faster_rcnn": {
                    "format": "coco",
                    "output_dir": "faster_rcnn"
                }
            }
        }
    })


@pytest.fixture
def mock_bdd100k_annotations():
    """ Mock BDD100K annotations. """
    return [
        {
            "name": "frame1.jpg",
            "videoName": "video1",
            "labels": [
                {
                    "category": "pedestrian",
                    "box2d": {"x1": 100, "y1": 56.25, "x2": 150, "y2": 106.25}
                },
                {
                    "category": "car",
                    "box2d": {"x1": 300, "y1": 168.75, "x2": 400, "y2": 268.75}
                }
            ]
        }
    ]


@pytest.fixture
def mock_bdd100k_frame():
    """ Mock BDD100K frame. """
    frame = Mock()
    frame.shape = (640, 640, 3)
    return frame
#endregion

#region DatasetConverter
class TestDatasetConverter(DatasetConverter):
    def convert_annotations(self, dataset_dir: str | Path, cfg: DictConfig) -> None:
        pass

    def process_dataset(self, dataset_dir: str | Path, cfg: DictConfig) -> None:
        pass


@patch("src.datasets.dataset_converter.log")
def test_convert_and_process_success(mock_hydra_config, temp_dir):
    """ Test successful execution of convert_and_process method with correct calls. """
    dataset_dir = temp_dir / "test_dataset"
    converter = TestDatasetConverter()

    # Verify results
    with patch.object(converter, "process_dataset") as mock_process, \
         patch.object(converter, "convert_annotations") as mock_convert:
        converter.convert_and_process(dataset_dir, mock_hydra_config)

        mock_process.assert_called_once_with(dataset_dir, mock_hydra_config)
        mock_convert.assert_called_once_with(dataset_dir, mock_hydra_config)


@patch("src.datasets.dataset_converter.log")
def test_convert_and_process_string_path_success(mock_hydra_config, temp_dir):
    """ Test convert_and_process with string path conversion to Path object. """
    dataset_dir = str(temp_dir / "test_dataset")
    converter = TestDatasetConverter()

    # Verify results
    with patch.object(converter, "process_dataset") as mock_process, \
         patch.object(converter, "convert_annotations") as mock_convert:
        converter.convert_and_process(dataset_dir, mock_hydra_config)

        mock_process.assert_called_once_with(Path(dataset_dir), mock_hydra_config)
        mock_convert.assert_called_once_with(Path(dataset_dir), mock_hydra_config)
#endregion

#region COCOConverter
def test_coco_converter_process_dataset_verify_no_processing(mock_coco_hydra_config):
    """ Test that COCOConverter process_dataset executes without processing methods. """
    converter = COCOConverter()

    # Verify results
    with patch("src.datasets.dataset_converter.log.info") as mock_log:
        converter.process_dataset("temporary_dir", mock_coco_hydra_config)
        mock_log.assert_called_once_with("No process methods for COCO dataset")


def test_coco_converter_convert_annotations_success(mock_coco_hydra_config, temp_dir):
    """ Test successful conversion of COCO annotations for all model formats. """
    converter = COCOConverter()
    dataset_dir = temp_dir

    # Verify results
    with patch.object(converter, "_convert_to_yolo") as mock_yolo, \
         patch.object(converter, "_convert_to_coco") as mock_coco:
        converter.convert_annotations(dataset_dir, mock_coco_hydra_config)

        mock_yolo.assert_called_once_with(dataset_dir, mock_coco_hydra_config, dataset_dir / "yolo")
        mock_coco.assert_called_once_with(dataset_dir, mock_coco_hydra_config, dataset_dir / "faster_rcnn")


def test_coco_converter_convert_annotations_invalid_format(mock_coco_hydra_config):
    """ Test that COCOConverter convert_annotations raises ValueError for invalid format. """
    invalid_config = DictConfig({
        "dataset": {
            "name": "coco",
            "model": {
                "invalid": {"format": "invalid", "output_dir": "invalid"}
            }
        }
    })
    converter = COCOConverter()

    # Verify results
    with pytest.raises(ValueError, match="Unknown format invalid"):
        converter.convert_annotations("temporary_dir", invalid_config)


@patch("builtins.open", new_callable=mock_open)
@patch("json.load")
@patch("PIL.Image.open")
def test_coco_converter_convert_to_yolo_success(mock_image_open, mock_json_load, mock_open_file, mock_coco_hydra_config, mock_coco_annotations, temp_dir):
    """ Test successful conversion of COCO annotations to YOLO format for train and val splits. """
    mock_json_load.return_value = mock_coco_annotations
    mock_image = Mock()
    mock_image.size = (640, 640)
    mock_image_open.return_value.__enter__.return_value = mock_image

    converter = COCOConverter()
    dataset_dir = temp_dir
    output_dir = dataset_dir / "yolo"

    for split in ["train", "val"]:
        annotations_dir = dataset_dir / "annotations"
        images_dir = dataset_dir / f"{split}2017"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        for image in mock_coco_annotations["images"]:
            image_path = images_dir / image["file_name"]
            with open(image_path, "wb") as f:
                f.write(b"")
        annotation_path = annotations_dir / f"instances_{split}2017.json"
        annotation_path.write_text(json.dumps(mock_coco_annotations))

    # Verify results
    with patch("pathlib.Path.exists", return_value=True):
        converter._convert_to_yolo(dataset_dir, mock_coco_hydra_config, output_dir)

    assert any(str(call[0][0]).replace('\\', '/').endswith(f"annotations/instances_{split}2017.json")
        and call[0][1] == "r" for split in ["train", "val"] for call in mock_open_file.call_args_list), \
        f"Expected annotations JSON to be opened for both 'train' and 'val', got: {[c[0] for c in mock_open_file.call_args_list]}"
    assert any(str(call[0][0]).replace('\\', '/').endswith(f"yolo/{split}/labels/000000000001.txt")
        and call[0][1] == "w" for split in ["train", "val"] for call in mock_open_file.call_args_list), \
        f"Expected YOLO label files to be created for both 'train' and 'val', got: {[c[0] for c in mock_open_file.call_args_list]}"
    handle = mock_open_file()
    handle.write.assert_any_call("0 0.195312 0.195312 0.078125 0.078125\n1 0.390625 0.390625 0.156250 0.156250"), \
        "Expected YOLO annotations to be written correctly."
    assert mock_image_open.call_count == 2, f"Expected 2 images to be opened, got {mock_image_open.call_count}"


@patch("builtins.open", new_callable=mock_open)
@patch("json.load")
def test_coco_converter_convert_to_coco_success(mock_json_load, mock_open_file, mock_coco_hydra_config, mock_coco_annotations, temp_dir):
    """ Test successful conversion of COCO annotations to COCO format for Faster-RCNN. """
    mock_json_load.return_value = mock_coco_annotations

    converter = COCOConverter()
    dataset_dir = temp_dir
    output_dir = dataset_dir / "faster_rcnn"

    for split in ["train", "val"]:
        annotations_dir = dataset_dir / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        annotation_path = annotations_dir / f"instances_{split}2017.json"
        annotation_path.write_text(json.dumps(mock_coco_annotations))

    converter._convert_to_coco(dataset_dir, mock_coco_hydra_config, output_dir)

    # Verify results
    written_files = [str(call[0][0]) for call in mock_open_file.call_args_list if call[0][1] == "w"]
    assert any(f"instances_{split}.json" in f for split in ["train", "val"] for f in written_files), f"Expected write calls for instances_{split}.json not found in {written_files}"


def test_coco_converter_convert_to_yolo_missing_annotations(mock_coco_hydra_config, temp_dir):
    """ Test that COCOConverter convert_to_yolo raises InvalidInputError for missing annotations. """
    converter = COCOConverter()
    dataset_dir = temp_dir
    for split in ["train", "val"]:
        (dataset_dir / f"{split}2017").mkdir(parents=True, exist_ok=True)

    # Verify results
    with pytest.raises(InvalidInputError, match="No annotations found for train dataset"):
        converter._convert_to_yolo(dataset_dir, mock_coco_hydra_config, temp_dir / "yolo")


@patch("builtins.open", new_callable=mock_open)
@patch("json.load")
def test_coco_converter_convert_to_coco_invalid_annotations(mock_json_load, mock_open_file, mock_coco_hydra_config, temp_dir):
    """ Test that COCOConverter convert_to_coco handles empty annotations correctly. """
    mock_json_load.return_value = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "car"}
        ]
    }

    converter = COCOConverter()
    dataset_dir = temp_dir
    output_dir = dataset_dir / "faster_rcnn"

    for split in ["train", "val"]:
        annotations_dir = dataset_dir / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        annotation_path = annotations_dir / f"instances_{split}2017.json"
        annotation_path.write_text(json.dumps(mock_json_load.return_value))

    converter._convert_to_coco(dataset_dir, mock_coco_hydra_config, output_dir)

    # Verify results
    written_files = [str(call[0][0]) for call in mock_open_file.call_args_list if call[0][1] == "w"]
    assert any(f"instances_{split}.json" in f for split in ["train", "val"] for f in written_files), f"Expected write calls for instances_{split}.json not found in {written_files}"
    handle = mock_open_file()
    handle.write.assert_called()
#endregion

#region BDD100KConverter
def test_bdd100k_converter_process_dataset_success(mock_bdd100k_hydra_config, temp_dir):
    """ Test successful processing of BDD100K dataset with resizing and video creation. """
    converter = BDD100KConverter()
    target_size = tuple[int, int](mock_bdd100k_hydra_config.dataset.image_target_size)
    dataset_dir = temp_dir
    (dataset_dir / "images/train").mkdir(parents=True)
    (dataset_dir / "images/val").mkdir()

    # Verify results
    with patch.object(converter, "_resize_images_in_folders") as mock_resize, \
            patch.object(converter, "_create_videos") as mock_videos:
        converter.process_dataset(dataset_dir, mock_bdd100k_hydra_config)

    mock_resize.assert_any_call(dataset_dir / "images/train", "train", target_size)
    mock_resize.assert_any_call(dataset_dir / "images/val", "val", target_size)
    mock_videos.assert_called_once_with(dataset_dir / "images/val", dataset_dir / "videos/val", mock_bdd100k_hydra_config.dataset.video_fps)


def test_bdd100k_converter_convert_annotations_success(mock_bdd100k_hydra_config, temp_dir):
    """ Test successful conversion of BDD100K annotations for all model formats. """
    converter = BDD100KConverter()
    dataset_dir = temp_dir

    # Verify results
    with patch.object(converter, "_convert_to_yolo") as mock_yolo, \
            patch.object(converter, "_convert_to_coco") as mock_coco:
        converter.convert_annotations(dataset_dir, mock_bdd100k_hydra_config)

    mock_yolo.assert_called_once_with(dataset_dir, mock_bdd100k_hydra_config, dataset_dir / mock_bdd100k_hydra_config.dataset.model.yolo.output_dir)
    mock_coco.assert_called_once_with(dataset_dir, mock_bdd100k_hydra_config, dataset_dir / mock_bdd100k_hydra_config.dataset.model.faster_rcnn.output_dir)


def test_bdd100k_converter_convert_annotations_invalid_class(mock_bdd100k_hydra_config):
    """ Test BDD100KConverter convert_annotations raises ValueError for invalid classes. """
    invalid_config = DictConfig({
        "dataset": {
            "name": "bdd100k",
            "classes": {"selected_classes": ["invalid_class"]},
            "model": {"yolo": {"format": "yolo", "output_dir": "yolo"}}
        }
    })
    converter = BDD100KConverter()

    # Verify results
    with pytest.raises(ValueError, match="Invalid selected classes: \\['invalid_class'\\]"):
        converter.convert_annotations("temporary_dir", invalid_config)


@patch("builtins.open", new_callable=mock_open)
@patch("json.load")
@patch("PIL.Image.open")
@patch("src.datasets.dataset_converter.log")
def test_bdd100k_converter_convert_to_coco_success(mock_log, mock_image_open, mock_json_load, mock_open_file, mock_bdd100k_hydra_config, mock_bdd100k_annotations, temp_dir):
    """ Test successful conversion and saving of BDD100K annotations to COCO format for train and val splits. """
    mock_json_load.return_value = mock_bdd100k_annotations
    mock_image = Mock()
    mock_image.size = tuple(mock_bdd100k_hydra_config.dataset.image_target_size)
    mock_image_open.return_value.__enter__.return_value = mock_image
    converter = BDD100KConverter()
    dataset_dir = temp_dir
    output_dir = temp_dir / mock_bdd100k_hydra_config.dataset.model.faster_rcnn.output_dir

    # Create directories and JSON files
    for split in ["train", "val"]:
        (dataset_dir / f"images/{split}/video1").mkdir(parents=True)
        (dataset_dir / f"annotations/{split}").mkdir(parents=True)
        (dataset_dir / f"images/{split}/video1/frame1.jpg").write_bytes(b"")
        (dataset_dir / f"annotations/{split}/video1.json").write_text(json.dumps(mock_bdd100k_annotations))

    # open_side_effect for reading JSON and writing COCO files
    written_files_content = {}
    def open_side_effect(file, mode='r'):
        path = Path(file).as_posix()
        if 'r' in mode and path.endswith(".json") and "annotations" in path:
            mock = mock_open(read_data=json.dumps(mock_bdd100k_annotations))()
            mock.__enter__.return_value = mock
            return mock
        if 'w' in mode and "faster_rcnn" in path:
            buffer = []
            mock = mock_open()()
            def write_mock(write_data):
                buffer.append(write_data)
                return len(write_data)
            def enter_mock():
                return mock
            def exit_mock(*_):
                written_files_content[path] = "".join(buffer)
            mock.write.side_effect = write_mock
            mock.__enter__.side_effect = enter_mock
            mock.__exit__.side_effect = exit_mock
            return mock
        return mock_open()

    mock_open_file.side_effect = open_side_effect

    # Scaling from 1280x720 to 640x640:
    # Expected COCO data structure with rounded values
    def round_bbox_area(bbox, area, d=2):
        return [round(x, d) for x in bbox], round(area, d)
    # pedestrian: x1=100/1280*640=50.0, y1=56.25/720*640=50.0, w=(150-100)/1280*640=25.0, h=(106.25-56.25)/720*640=44.44444444444444
    # car: x1=300/1280*640=150.0, y1=168.75/720*640=150.0, w=(400-300)/1280*640=50.0, h=(268.75-168.75)/720*640=88.88888888888889
    bbox1, area1 = round_bbox_area([50.0, 50.0, 25.0, 44.44444444444444], 1111.111111111111)
    bbox2, area2 = round_bbox_area([150.0, 150.0, 50.0, 88.88888888888889], 4444.444444444444)
    expected_coco_data = {
        "images": [{"id": 0, "file_name": "video1/frame1.jpg", "width": 640, "height": 640}],
        "annotations": [
            {"id": 0, "image_id": 0, "category_id": 1, "bbox": bbox1, "area": area1, "iscrowd": 0},
            {"id": 1, "image_id": 0, "category_id": 3, "bbox": bbox2, "area": area2, "iscrowd": 0}
        ],
        "categories": converter._COCO_CATEGORIES
    }

    # Verify results
    # Verify paths
    with patch("pathlib.Path.exists", return_value=True):
        for split in ["train", "val"]:
            json_dir = dataset_dir / f"annotations/{split}"
            with patch("pathlib.Path.glob", return_value=[json_dir / "video1.json"]):
                converter._convert_to_coco(dataset_dir, mock_bdd100k_hydra_config, output_dir)

    # Verify written COCO data
    for split in ["train", "val"]:
        write_path = (output_dir / f"{split}.json").as_posix()
        assert write_path in written_files_content, f"No output for {split}"
        data = json.loads(written_files_content[write_path])
        for ann in data["annotations"]:
            ann["bbox"] = [round(x, 2) for x in ann["bbox"]]
            ann["area"] = round(ann["area"], 2)
        assert data == expected_coco_data, f"Written COCO data for {split} does not match expected: got {data}"

    # Verify logs
    mock_log.info.assert_any_call("[TRAIN] Processing 1 JSON files")
    mock_log.info.assert_any_call("[VAL] Processing 1 JSON files")
    mock_log.info.assert_any_call("[TRAIN] Created 1 images, 2 annotations, 0 skipped bboxes, 0 empty frames")
    mock_log.info.assert_any_call("[VAL] Created 1 images, 2 annotations, 0 skipped bboxes, 0 empty frames")


@patch("src.datasets.dataset_converter.log")
def test_bdd100k_converter_convert_to_coco_nonexistent_dir(mock_log, mock_bdd100k_hydra_config, temp_dir):
    """ Test convert_to_coco raising InvalidInputError for nonexistent json_dir or image_dir. """
    converter = BDD100KConverter()
    dataset_dir = temp_dir
    output_dir = temp_dir / "coco"

    # Verify results
    # Nonexistent json directory
    with patch("pathlib.Path.exists", autospec=True, side_effect=lambda self: False):
        try:
            converter._convert_to_coco(dataset_dir, mock_bdd100k_hydra_config, output_dir)
            assert False, "Expected InvalidInputError for nonexistent json_dir"
        except InvalidInputError as e:
            assert str(e) == f"Annotations directory {dataset_dir / 'annotations/train'} does not exist",  f"Unexpected error message: {str(e)}"
        mock_log.error.assert_any_call(f"Annotations directory {dataset_dir / 'annotations/train'} does not exist")

    # Nonexistent image directory
    with patch("pathlib.Path.exists", autospec=True, side_effect=lambda path_self: "annotations" in str(path_self)):
        try:
            converter._convert_to_coco(dataset_dir, mock_bdd100k_hydra_config, output_dir)
            assert False, "Expected InvalidInputError for nonexistent image_dir"
        except InvalidInputError as e:
            assert str(e) == f"Images directory {dataset_dir / 'images/train'} does not exist", f"Unexpected error message: {str(e)}"
        mock_log.error.assert_any_call(f"Images directory {dataset_dir / 'images/train'} does not exist")


@patch("json.load")
@patch("PIL.Image.open")
def test_bdd100k_converter_process_frame_coco_success(mock_image_open, mock_json_load, mock_bdd100k_hydra_config, mock_bdd100k_annotations, temp_dir):
    """ Test successful processing of a single frame for COCO format in BDD100KConverter. """
    mock_json_load.return_value = mock_bdd100k_annotations
    mock_image = Mock()
    mock_image.size = tuple(mock_bdd100k_hydra_config.dataset.image_target_size)
    mock_image_open.return_value.__enter__.return_value = mock_image

    converter = BDD100KConverter()
    dataset_dir = temp_dir
    image_dir = dataset_dir / "images/train/video1"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "frame1.jpg"
    image_path.write_bytes(b"")

    # Verify results
    with patch("pathlib.Path.glob", return_value=[image_path]):
        image_data, annotations, is_empty, skipped_boxes, skipped_categories = converter._process_frame(
            mock_bdd100k_annotations[0], dataset_dir / "images/train", 640, 640, ["person", "car"], "coco", "video1"
        )

    assert image_data["file_name"] == "video1/frame1.jpg", f"Unexpected file_name: {image_data['file_name']}"
    assert len(annotations) == 2, f"Expected 2 annotations, got {len(annotations)}"
    assert annotations[0]["category_id"] == 1, f"Expected category_id=1 for first annotation, got {annotations[0]['category_id']}"
    assert annotations[1]["category_id"] == 3, f"Expected category_id=3 for second annotation, got {annotations[1]['category_id']}"
    assert is_empty is False, "Expected is_empty to be False"
    assert skipped_boxes == 0, f"Expected 0 skipped boxes, got {skipped_boxes}"
    assert len(skipped_categories) == 0, f"Expected no skipped categories, got {skipped_categories}"


@patch("PIL.Image.open")
@patch("src.datasets.dataset_converter.log")
def test_bdd100k_converter_resize_images_success(mock_log, mock_image_open, mock_bdd100k_hydra_config, temp_dir):
    """ Test successful resizing of images in BDD100KConverter. """
    mock_image = Mock()
    mock_image.resize = Mock(return_value=mock_image)
    mock_image_open.return_value.__enter__.return_value = mock_image

    converter = BDD100KConverter()
    target_size = tuple[int, int](mock_bdd100k_hydra_config.dataset.image_target_size)
    images_path = temp_dir / "images/train"
    video1_path = images_path / "video1"
    video1_path.mkdir(parents=True)
    image_path = video1_path / "image1.jpg"
    image_path.write_bytes(b"")

    # Verify results
    with patch("pathlib.Path.iterdir", return_value=[video1_path]):
        with patch("pathlib.Path.glob", return_value=[image_path]):
            converter._resize_images_in_folders(images_path, "train", target_size)

    mock_image_open.assert_called_once_with(image_path)
    mock_image.resize.assert_called_once_with(target_size, Image.Resampling.LANCZOS)
    mock_image.save.assert_called_once_with(image_path, quality=95)
    mock_log.info.assert_any_call("[TRAIN] (1) Finished processing folder: video1")


@patch("builtins.open", new_callable=mock_open)
@patch("json.load")
@patch("PIL.Image.open")
@patch("src.datasets.dataset_converter.log")
def test_bdd100k_converter_convert_to_yolo_success(mock_log, mock_image_open, mock_json_load, mock_open_file, mock_bdd100k_hydra_config, mock_bdd100k_annotations, temp_dir):
    """ Test successful conversion and saving of BDD100K annotations to YOLO format for train and val splits. """
    mock_json_load.return_value = mock_bdd100k_annotations
    mock_image = Mock()
    mock_image.size = tuple(mock_bdd100k_hydra_config.dataset.image_target_size)
    mock_image_open.return_value.__enter__.return_value = mock_image
    converter = BDD100KConverter()
    dataset_dir = temp_dir
    output_dir = temp_dir / mock_bdd100k_hydra_config.dataset.model.yolo.output_dir

    # Create directories and JSON files
    for split in ["train", "val"]:
        (dataset_dir / f"images/{split}/video1").mkdir(parents=True)
        (dataset_dir / f"annotations/{split}").mkdir(parents=True)
        (dataset_dir / f"images/{split}/video1/frame1.jpg").write_bytes(b"")
        (dataset_dir / f"annotations/{split}/video1.json").write_text(json.dumps(mock_bdd100k_annotations))

    # open_side_effect for reading JSON and writing YOLO files
    written_files_content = {}
    def open_side_effect(file, mode='r'):
        path = Path(file).as_posix()
        if 'r' in mode and "annotations" in path and path.endswith(".json"):
            mock = mock_open(read_data=json.dumps(mock_bdd100k_annotations))()
            mock.__enter__.return_value = mock
            return mock
        if 'w' in mode and "yolo" in path and path.endswith(".txt"):
            buffer = []
            mock = mock_open()()
            def write_mock(data):
                buffer.append(data)
                return len(data)
            def enter_mock():
                return mock
            def exit_mock(*_):
                written_files_content[path] = "".join(buffer)
            mock.write.side_effect = write_mock
            mock.__enter__.side_effect = enter_mock
            mock.__exit__.side_effect = exit_mock
            return mock
        return mock_open()()

    mock_open_file.side_effect = open_side_effect

    # Expected YOLO annotations (normalized center_x, center_y, width, height)
    # Scaling from 1280x720 to 640x640 (normalized to 0-1):
    # pedestrian: x1=100/1280*640=50.0, y1=56.25/720*640=50.0, x2=150/1280*640=75.0, y2=106.25/720*640=94.44444444444444
    # width_bbox=25.0, height_bbox=44.44444444444444
    # x_center=(50.0+25.0/2)/640=0.09765625, y_center=(50.0+44.44444444444444/2)/640=0.11284722222222222
    # width_norm=25.0/640=0.0390625, height_norm=44.44444444444444/640=0.06944444444444445
    # car: x1=300/1280*640=150.0, y1=168.75/720*640=150.0, x2=400/1280*640=200.0, y2=268.75/720*640=238.88888888888889
    # width_bbox=50.0, height_bbox=88.88888888888889
    # x_center=(150.0+50.0/2)/640=0.2734375, y_center=(150.0+88.88888888888889/2)/640=0.30381944444444444
    # width_norm=50.0/640=0.078125, height_norm=88.88888888888889/640=0.1388888888888889
    expected_yolo_data = (
        "0 0.097656 0.112847 0.039062 0.069444\n"
        "2 0.273438 0.303819 0.078125 0.138889"
    )

    # Verify results
    # Verify paths
    with patch("pathlib.Path.exists", return_value=True):
        for split in ["train", "val"]:
            json_dir = dataset_dir / f"annotations/{split}"
            with patch("pathlib.Path.glob", return_value=[json_dir / "video1.json"]):
                converter._convert_to_yolo(dataset_dir, mock_bdd100k_hydra_config, output_dir)

    # Verify written YOLO data
    for split in ["train", "val"]:
        write_path = (output_dir / split / "video1" / "frame1.txt").as_posix()
        assert write_path in written_files_content, f"Missing expected YOLO output file: {write_path}"
        written_data = written_files_content[write_path]
        assert written_data == expected_yolo_data, f"Incorrect YOLO annotation content in {write_path}. Got:\n{written_data}\nExpected:\n{expected_yolo_data}"

    # Verify logs
    mock_log.info.assert_any_call("[TRAIN] Processing 1 JSON files")
    mock_log.info.assert_any_call("[VAL] Processing 1 JSON files")
    mock_log.info.assert_any_call("[TRAIN] Created 2 annotations, 0 empty frames, 0 skipped bboxes")
    mock_log.info.assert_any_call("[VAL] Created 2 annotations, 0 empty frames, 0 skipped bboxes")


@patch("PIL.Image.open")
def test_bdd100k_converter_process_frame_invalid_category(mock_image_open, mock_bdd100k_hydra_config, temp_dir):
    """ Test _process_frame skipping annotations with invalid categories. """
    mock_image = Mock()
    mock_image.size = tuple[int, int](mock_bdd100k_hydra_config.dataset.image_target_size)
    mock_image_open.return_value.__enter__.return_value = mock_image
    converter = BDD100KConverter()
    dataset_dir = temp_dir
    image_dir = dataset_dir / "images/train/video1"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "frame1.jpg"
    image_path.write_bytes(b"")

    # Mock frame with valid and invalid categories
    frame = {
        "name": "frame1.jpg",
        "videoName": "video1",
        "labels": [
            {"category": "pedestrian", "box2d": {"x1": 100, "y1": 56.25, "x2": 150, "y2": 106.25}},  # maps to "person"
            {"category": "truck", "box2d": {"x1": 200, "y1": 156.25, "x2": 250, "y2": 206.25}},     # invalid class for selected_classes
            {"category": "car", "box2d": {"x1": 300, "y1": 168.75, "x2": 400, "y2": 268.75}}        # valid
        ]
    }

    # Expected YOLO annotations (only for valid categories: person, car)
    expected_yolo_annotations = [
        "0 0.097656 0.112847 0.039062 0.069444",
        "2 0.273438 0.303819 0.078125 0.138889"
    ]

    # Verify results
    for format_type in ["coco", "yolo"]:
        with patch("pathlib.Path.glob", return_value=[image_path]):
            image_data, annotations, is_empty, skipped_boxes, skipped_categories = converter._process_frame(
                frame, dataset_dir / "images/train", 640, 640, ["person", "car"], format_type, "video1"
            )

        expected_image_data = {
            "file_name": "video1/frame1.jpg",
            "width": 640,
            "height": 640
        } if format_type == "coco" else {
            "txt_path": dataset_dir / "images/train/video1/frame1.txt"
        }

        assert image_data == expected_image_data, f"Image data mismatch for format '{format_type}': expected {expected_image_data}, got {image_data}"
        assert is_empty is False, f"Expected frame is_empty to be False for format '{format_type}', got {is_empty}"
        assert skipped_boxes == 0, f"Expected no skipped boxes for format '{format_type}', got {skipped_boxes}"
        assert skipped_categories == {"truck"}, f"Expected skipped categories to contain 'truck' for format '{format_type}', got {skipped_categories}"

        if format_type == "yolo":
            assert annotations == expected_yolo_annotations, f"YOLO annotations mismatch: expected {expected_yolo_annotations}, got {annotations}"
        else:
            assert len(annotations) == 2, f"Expected 2 COCO annotations, got {len(annotations)}"
            assert annotations[0]["category_id"] == 1, f"Expected first COCO annotation category_id 1 (person), got {annotations[0]['category_id']}"
            assert annotations[1]["category_id"] == 3, f"Expected second COCO annotation category_id 3 (car), got {annotations[1]['category_id']}"
            assert annotations[0]["bbox"] == [50.0, 50.0, 25.0, 44.44444444444444], f"COCO bbox for first annotation mismatch: {annotations[0]['bbox']}"
            assert annotations[1]["bbox"] == [150.0, 150.0, 50.0, 88.88888888888889], f"COCO bbox for second annotation mismatch: {annotations[1]['bbox']}"


@patch("PIL.Image.open")
def test_bdd100k_converter_process_frame_empty_frame(mock_image_open, mock_bdd100k_hydra_config, temp_dir):
    """ Test processing of an empty frame (no labels) for COCO and YOLO formats in BDD100KConverter. """
    mock_image = Mock()
    mock_image.size = tuple(mock_bdd100k_hydra_config.dataset.image_target_size)
    mock_image_open.return_value.__enter__.return_value = mock_image
    converter = BDD100KConverter()
    dataset_dir = temp_dir
    image_dir = dataset_dir / "images/train/video1"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "frame1.jpg"
    image_path.write_bytes(b"")

    # Mock empty frame annotations
    empty_frame = {
        "name": "frame1.jpg",
        "videoName": "video1",
        "labels": []
    }

    # Verify results
    for format_type in ["coco", "yolo"]:
        with patch("pathlib.Path.glob", return_value=[image_path]):
            image_data, annotations, is_empty, skipped_boxes, skipped_categories = converter._process_frame(
                empty_frame, dataset_dir / "images/train", 640, 640, ["person", "car"], format_type, "video1"
            )

        expected_image_data = {
            "file_name": "video1/frame1.jpg",
            "width": 640,
            "height": 640
        } if format_type == "coco" else {
            "txt_path": dataset_dir / "images/train/video1/frame1.txt"
        }

        assert image_data == expected_image_data, f"Image data mismatch for empty frame with format '{format_type}': expected {expected_image_data}, got {image_data}"
        assert annotations == [], f"Expected no annotations for empty frame with format '{format_type}', got {annotations}"
        assert is_empty is True, f"Expected is_empty True for empty frame with format '{format_type}', got {is_empty}"
        assert skipped_boxes == 0, f"Expected no skipped boxes for empty frame with format '{format_type}', got {skipped_boxes}"
        assert skipped_categories == set(), f"Expected no skipped categories for empty frame with format '{format_type}', got {skipped_categories}"


@patch("src.datasets.dataset_converter.log")
@patch("json.load")
def test_bdd100k_converter_convert_to_yolo_invalid_json(mock_json_load, mock_log, mock_bdd100k_hydra_config, temp_dir):
    """ Test convert_to_yolo logs error when JSON is invalid. """
    converter = BDD100KConverter()
    dataset_dir = temp_dir
    output_dir = temp_dir / mock_bdd100k_hydra_config.dataset.model["yolo"]["output_dir"]
    mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)

    for split in ["train", "val"]:
        images_dir = dataset_dir / f"images/{split}/video1"
        annotations_dir = dataset_dir / f"annotations/{split}"
        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)
        (images_dir / "frame1.jpg").write_bytes(b"")
        (annotations_dir / "video1.json").write_text("INVALID JSON")

    # open_side_effect handle reading JSON and images
    def open_side_effect(file, mode='r'):
        file_path = Path(file)
        if "annotations" in str(file_path) and mode == "r":
            mock = mock_open(read_data="INVALID JSON").return_value
            mock.__iter__.return_value = ["INVALID JSON"]
            return mock
        elif "images" in str(file_path) and mode == "rb":
            mock = mock_open(read_data=b"").return_value
            return mock
        else:
            return mock_open().return_value

    mock_open.side_effect = open_side_effect

    # Verify results
    # Verify paths
    with patch("pathlib.Path.exists", return_value=True):
        original_glob = Path.glob

        def glob_side_effect(self, pattern):
            if "annotations" in str(self) and pattern == "*.json":
                return [self / "video1.json"]
            else:
                return original_glob(self, pattern)

        with patch("pathlib.Path.glob", new=glob_side_effect):
            converter._convert_to_yolo(dataset_dir, mock_bdd100k_hydra_config, output_dir)

    # Verify no .txt created for invalid JSON
    for split in ["train", "val"]:
        txt_dir = output_dir / split / "video1"
        if txt_dir.exists():
            txt_files = list(txt_dir.glob("*.txt"))
            assert not txt_files, f"Expected no .txt files created in {txt_dir} for invalid JSON input, but found: {txt_files}"

    for split in ["train", "val"]:
        json_file = str(dataset_dir / f"annotations/{split}/video1.json")
        error_found = any(
            call.args and
            isinstance(call.args[0], str) and
            f"Failed to read JSON {json_file}:" in call.args[0] and
            "Invalid JSON" in call.args[0]
            for call in mock_log.error.call_args_list
        )
        assert error_found, f"Expected error log about invalid JSON for file {json_file} not found in logs"


@patch("json.load")
@patch("PIL.Image.open")
def test_bdd100k_converter_process_frame_yolo_success(mock_image_open, mock_json_load, mock_bdd100k_hydra_config, mock_bdd100k_annotations, temp_dir):
    """ Test successful processing of a single frame for YOLO format in BDD100KConverter. """
    mock_json_load.return_value = mock_bdd100k_annotations
    mock_image = Mock()
    mock_image.size = tuple(mock_bdd100k_hydra_config.dataset.image_target_size)
    mock_image_open.return_value.__enter__.return_value = mock_image

    converter = BDD100KConverter()
    dataset_dir = temp_dir
    image_dir = dataset_dir / "images/train/video1"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "frame1.jpg"
    image_path.write_bytes(b"")

    # Verify results
    with patch("pathlib.Path.glob", return_value=[image_path]):
        image_data, annotations, is_empty, skipped_boxes, skipped_categories = converter._process_frame(
            mock_bdd100k_annotations[0], dataset_dir / "images/train", 640, 640, ["person", "car"], "yolo", "video1"
        )

    assert image_data["txt_path"] == dataset_dir / "images/train/video1/frame1.txt", f"Expected txt_path to be '{dataset_dir / 'images/train/video1/frame1.txt'}', got {image_data['txt_path']}"
    assert len(annotations) == 2, f"Expected 2 annotations, got {len(annotations)}"
    assert annotations[0].startswith("0 "), f"Expected first annotation to start with '0 ' (person), got '{annotations[0]}'"
    assert annotations[1].startswith("2 "), f"Expected second annotation to start with '2 ' (car), got '{annotations[1]}'"
    assert is_empty is False, f"Expected is_empty to be False, got {is_empty}"
    assert skipped_boxes == 0, f"Expected skipped_boxes to be 0, got {skipped_boxes}"
    assert len(skipped_categories) == 0, f"Expected no skipped_categories, got {skipped_categories}"


@patch("cv2.imread")
@patch("cv2.VideoWriter")
@patch("cv2.VideoWriter_fourcc")
def test_bdd100k_converter_create_videos_success(mock_fourcc, mock_video_writer, mock_imread, mock_bdd100k_frame, mock_bdd100k_hydra_config, temp_dir):
    """ Test successful creation of videos from frames in BDD100KConverter. """
    mock_bdd100k_frame.shape = (640, 640, 3)
    mock_imread.return_value = mock_bdd100k_frame
    mock_writer = Mock()
    mock_video_writer.return_value = mock_writer
    mock_fourcc.return_value = "FAKE_FOURCC"

    converter = BDD100KConverter()
    images_path = temp_dir / "images/val/video1"
    videos_path = temp_dir / "videos/val"
    images_path.mkdir(parents=True)
    image_path1 = images_path / "0000001.jpg"
    image_path2 = images_path / "0000002.jpg"
    image_path1.write_bytes(b"")
    image_path2.write_bytes(b"")

    # Verify results
    with patch("pathlib.Path.glob", return_value=[image_path1, image_path2]):
        converter._create_videos(images_path.parent, videos_path, mock_bdd100k_hydra_config.dataset.video_fps)

    mock_fourcc.assert_called_once_with(*"mp4v")
    mock_video_writer.assert_called_once_with(
        str(videos_path / "video1.mp4"), "FAKE_FOURCC", mock_bdd100k_hydra_config.dataset.video_fps, mock_bdd100k_hydra_config.dataset.image_target_size
    )
    assert mock_writer.write.call_count == 2, f"Expected 2 frames to be written, but write was called {mock_writer.write.call_count} times"

@patch("cv2.imread")
@patch("cv2.VideoWriter")
@patch("cv2.VideoWriter_fourcc")
def test_bdd100k_converter_create_videos_empty_folder(mock_fourcc, mock_video_writer, mock_imread, mock_bdd100k_hydra_config, temp_dir):
    """ Test create_videos behaviour with empty folder. """
    mock_imread.return_value = None
    mock_fourcc.return_value = "FAKE_FOURCC"

    converter = BDD100KConverter()
    images_path = temp_dir / "images" / "val" / "empty_video"
    videos_path = temp_dir / "videos" / "val"
    images_path.mkdir(parents=True)
    converter._create_videos(images_path.parent, videos_path, mock_bdd100k_hydra_config.dataset.video_fps)

    # Verify results
    assert mock_video_writer.call_count == 0, f"Expected no video to be created because the image folder '{images_path}' is empty"


@patch("cv2.imread")
@patch("cv2.VideoWriter")
@patch("cv2.VideoWriter_fourcc")
def test_bdd100k_converter_create_videos_skip_invalid_frames(mock_fourcc, mock_video_writer, mock_imread, mock_bdd100k_frame, mock_bdd100k_hydra_config, temp_dir):
    """ Test create_videos skip invalid frames successfully. """
    mock_imread.side_effect = lambda path: mock_bdd100k_frame if "0000001" in str(path) else None
    mock_writer = Mock()
    mock_video_writer.return_value = mock_writer
    mock_fourcc.return_value = "FAKE_FOURCC"

    converter = BDD100KConverter()
    images_path = temp_dir / "images" / "val" / "video_skip"
    videos_path = temp_dir / "videos" / "val"
    images_path.mkdir(parents=True)
    (images_path / "0000001.jpg").write_bytes(b"")
    (images_path / "0000002.jpg").write_bytes(b"")

    converter._create_videos(images_path.parent, videos_path, mock_bdd100k_hydra_config.dataset.video_fps)

    # Verify results
    assert mock_writer.write.call_count == 1, f"Expected only valid frames to be written, but got {mock_writer.write.call_count} calls"


@patch("cv2.imread")
@patch("cv2.VideoWriter")
@patch("cv2.VideoWriter_fourcc")
def test_bdd100k_converter_create_videos_frame_order(mock_fourcc, mock_video_writer, mock_imread, temp_dir, mock_bdd100k_frame, mock_bdd100k_hydra_config):
    """ Test that create_videos create video with correct order of frames. """
    mock_imread.side_effect = mock_imread.side_effect = lambda path: mock_bdd100k_frame
    mock_writer = Mock()
    mock_video_writer.return_value = mock_writer
    mock_fourcc.return_value = "FAKE_FOURCC"

    converter = BDD100KConverter()
    images_path = temp_dir / "images" / "val" / "videoX"
    videos_path = temp_dir / "videos" / "val"
    images_path.mkdir(parents=True)

    (images_path / "0000002.jpg").write_bytes(b"")
    (images_path / "0000001.jpg").write_bytes(b"")

    converter._create_videos(images_path.parent, videos_path, mock_bdd100k_hydra_config.dataset.video_fps)

    # Verify results
    called_paths = [Path(call.args[0]).name for call in mock_imread.call_args_list]
    assert called_paths == sorted(called_paths, key=lambda x: int(Path(x).stem)), (
        f"Frames were read in order {called_paths}, "
        f"but expected sorted order {sorted(called_paths, key=lambda x: int(Path(x).stem))}"
    )
#endregion