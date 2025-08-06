import pytest
from pathlib import Path
from unittest.mock import patch
from omegaconf import DictConfig

from src.datasets.dataset_manager import DatasetManager
from src.exceptions.exceptions import DownloadError, InvalidInputError, ExtractionError


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory."""
    return tmp_path


@pytest.fixture
def mock_hydra_config():
    """Mock Hydra configuration."""
    manager_config_download = DictConfig({
        "manager": {
            "dataset_dir": "datasets",
            "types": ["coco", "bdd100k"],
            "stage": 0  # Download stage
        }
    })
    manager_config_convert = DictConfig({
        "manager": {
            "dataset_dir": "datasets",
            "types": ["coco", "bdd100k"],
            "stage": 1  # Convert stage
        }
    })
    coco_config = DictConfig({
        "dataset": {
            "name": "coco",
            "download_urls": [
                {"url": "http://images.cocodataset.org/zips/train2017.zip", "file_name": "train2017.zip"},
                {"url": "http://images.cocodataset.org/zips/val2017.zip", "file_name": "val2017.zip"},
                {"url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "file_name": "annotations_trainval2017.zip"}
            ],
            "extract_dir": "coco",
            "download_timeout": 60,
            "chunk_size": 8192,
            "paths": {
                "train_images": "coco/train2017",
                "val_images": "coco/val2017",
                "train_annotations": "coco/annotations/instances_train2017.json",
                "val_annotations": "coco/annotations/instances_val2017.json"
            }
        }
    })
    bdd100k_config = DictConfig({
        "dataset": {
            "name": "bdd100k",
            "download_urls": [
                {"url": "http://dl.yf.io/bdd100k/video_parts/bdd100k_videos_train_00.zip", "file_name": "bdd100k_videos_train_00.zip"},
                {"url": "http://dl.yf.io/bdd100k/video_parts/bdd100k_videos_val_00.zip", "file_name": "bdd100k_videos_val_00.zip"},
                {"url": "http://dl.yf.io/bdd100k/bdd100k_labels.zip", "file_name": "bdd100k_labels.zip"}
            ],
            "extract_dir": "bdd100k",
            "download_timeout": 60,
            "chunk_size": 8192,
            "paths": {
                "train_images": "bdd100k/images/train",
                "val_images": "bdd100k/images/val",
                "train_annotations": "bdd100k/100k/train",
                "val_annotations": "bdd100k/100k/val"
            }
        }
    })
    dataset_configs = {"coco": coco_config, "bdd100k": bdd100k_config}
    return {
        "manager_download": manager_config_download,
        "manager_convert": manager_config_convert,
        "dataset_configs": dataset_configs}


@patch("src.datasets.dataset_manager.COCODownloader.download_and_extract")
@patch("src.datasets.dataset_manager.BDD100KDownloader.download_and_extract")
def test_dataset_manager_download_success(mock_bdd100k_download, mock_coco_download, mock_hydra_config):
    """ Test dataset manager success. """
    manager = DatasetManager()
    manager.run_manager(mock_hydra_config["manager_download"], mock_hydra_config["dataset_configs"])

    # Verify results
    mock_coco_download.assert_called_once_with(Path("datasets"), mock_hydra_config["dataset_configs"]["coco"])
    mock_bdd100k_download.assert_called_once_with(Path("datasets"), mock_hydra_config["dataset_configs"]["bdd100k"])


@patch("src.datasets.dataset_converter.COCOConverter.convert_and_process")
@patch("src.datasets.dataset_converter.BDD100KConverter.convert_and_process")
def test_dataset_manager_convert_success(mock_bdd100k_convert, mock_coco_convert, mock_hydra_config):
    """ Test dataset manager success for convert stage. """
    manager = DatasetManager()
    manager.run_manager(mock_hydra_config["manager_convert"], mock_hydra_config["dataset_configs"])

    mock_coco_convert.assert_called_once_with(Path("datasets"), mock_hydra_config["dataset_configs"]["coco"])
    mock_bdd100k_convert.assert_called_once_with(Path("datasets"), mock_hydra_config["dataset_configs"]["bdd100k"])


@patch("src.datasets.dataset_manager.COCODownloader.download_and_extract")
def test_dataset_manager_download_failure(mock_coco_download, mock_hydra_config):
    """ Test dataset manager download failure. """
    mock_coco_download.side_effect = DownloadError("Failed to download")
    manager = DatasetManager()

    # Verify results
    with pytest.raises(DownloadError):
        manager.run_manager(mock_hydra_config["manager_download"], mock_hydra_config["dataset_configs"])

    mock_coco_download.assert_called_once_with(Path("datasets"), mock_hydra_config["dataset_configs"]["coco"])


@patch("src.datasets.dataset_manager.COCODownloader.download_and_extract")
def test_dataset_manager_extraction_failure(mock_coco_download, mock_hydra_config):
    """ Test dataset manager extraction failure. """
    mock_coco_download.side_effect = ExtractionError("Failed to extract")
    manager = DatasetManager()

    # Verify results
    with pytest.raises(ExtractionError):
        manager.run_manager(mock_hydra_config["manager_download"], mock_hydra_config["dataset_configs"])

    mock_coco_download.assert_called_once_with(Path("datasets"), mock_hydra_config["dataset_configs"]["coco"])


@patch("src.datasets.dataset_manager.COCODownloader.download_and_extract")
def test_dataset_manager_invalid_input(mock_coco_download, mock_hydra_config):
    """ Test dataset manager invalid input. """
    mock_coco_download.side_effect = InvalidInputError("Invalid URL")
    manager = DatasetManager()

    # Verify results
    with pytest.raises(InvalidInputError):
        manager.run_manager(mock_hydra_config["manager_download"], mock_hydra_config["dataset_configs"])

    mock_coco_download.assert_called_once_with(Path("datasets"), mock_hydra_config["dataset_configs"]["coco"])


@patch("src.datasets.dataset_downloader.COCODownloader.download_and_extract")
@patch("src.datasets.dataset_downloader.BDD100KDownloader.download_and_extract")
def test_dataset_manager_partial_failure(mock_bdd100k_download, mock_coco_download, mock_hydra_config):
    """ Test dataset manager with one dataset failing. """
    mock_coco_download.side_effect = None
    mock_bdd100k_download.side_effect = DownloadError("Failed to download bdd100k")
    manager = DatasetManager()

    # Verify results
    with pytest.raises(DownloadError):
        manager.run_manager(mock_hydra_config["manager_download"], mock_hydra_config["dataset_configs"])

    mock_coco_download.assert_called_once_with(Path("datasets"), mock_hydra_config["dataset_configs"]["coco"])
    mock_bdd100k_download.assert_called_once_with(Path("datasets"), mock_hydra_config["dataset_configs"]["bdd100k"])

def test_dataset_manager_invalid_dataset_dir(mock_hydra_config):
    """ Test dataset manager with invalid dataset directory. """
    invalid_config = DictConfig({
        "manager": {
            "dataset_dir": "/invalid/path/123",
            "types": ["coco"],
            "stage": 0
        }
    })
    manager = DatasetManager()

    # Verify results
    with patch("src.datasets.dataset_downloader.COCODownloader.download_and_extract") as mock_coco_download:
        manager.run_manager(invalid_config, mock_hydra_config["dataset_configs"])
        mock_coco_download.assert_called_once_with(Path("/invalid/path/123"), mock_hydra_config["dataset_configs"]["coco"])

@patch("src.datasets.dataset_converter.COCOConverter.convert_and_process")
def test_dataset_manager_convert_failure(mock_coco_convert, mock_hydra_config):
    """ Test dataset manager conversion failure. """
    mock_coco_convert.side_effect = InvalidInputError("Invalid annotations")
    manager = DatasetManager()

    # Verify results
    with pytest.raises(InvalidInputError):
        manager.run_manager(mock_hydra_config["manager_convert"], mock_hydra_config["dataset_configs"])

    mock_coco_convert.assert_called_once_with(Path("datasets"), mock_hydra_config["dataset_configs"]["coco"])


def test_dataset_manager_invalid_type(mock_hydra_config):
    """ Test dataset manager invalid type. """
    invalid_config = DictConfig({
        "manager": {
            "dataset_dir": "datasets",
            "types": ["invalid"],
            "stage": 0
        }
    })
    manager = DatasetManager()

    # Verify results
    with pytest.raises(ValueError):
        manager.run_manager(invalid_config, mock_hydra_config["dataset_configs"])


def test_dataset_manager_invalid_stage(mock_hydra_config):
    """ Test dataset manager invalid stage. """
    invalid_config = DictConfig({
        "manager": {
            "dataset_dir": "datasets",
            "types": ["coco"],
            "stage": 200  # Invalid stage
        }
    })
    manager = DatasetManager()

    with pytest.raises(ValueError):
        manager.run_manager(invalid_config, mock_hydra_config["dataset_configs"])
