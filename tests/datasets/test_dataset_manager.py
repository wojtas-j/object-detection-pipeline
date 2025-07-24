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
    manager_config = DictConfig({
        "manager": {
            "dataset_dir": "datasets",
            "types": ["coco", "bdd100k"]
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
            "chunk_size": 8192
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
            "chunk_size": 8192
        }
    })
    dataset_configs = {"coco": coco_config, "bdd100k": bdd100k_config}
    return {"manager": manager_config, "dataset_configs": dataset_configs}


@patch("src.datasets.dataset_manager.COCODownloader.download_and_extract")
@patch("src.datasets.dataset_manager.BDD100KDownloader.download_and_extract")
def test_dataset_manager_success(mock_bdd100k_download, mock_coco_download, mock_hydra_config):
    """Test dataset manager success."""
    manager = DatasetManager()
    manager.run_manager(mock_hydra_config["manager"], mock_hydra_config["dataset_configs"])

    # Verify results
    mock_coco_download.assert_called_once_with(Path("datasets"), mock_hydra_config["dataset_configs"]["coco"])
    mock_bdd100k_download.assert_called_once_with(Path("datasets"), mock_hydra_config["dataset_configs"]["bdd100k"])


@patch("src.datasets.dataset_manager.COCODownloader.download_and_extract")
def test_dataset_manager_download_failure(mock_coco_download, mock_hydra_config):
    """Test dataset manager download failure."""
    mock_coco_download.side_effect = DownloadError("Failed to download")
    manager = DatasetManager()

    # Verify results
    with pytest.raises(DownloadError):
        manager.run_manager(mock_hydra_config["manager"], mock_hydra_config["dataset_configs"])

    mock_coco_download.assert_called_once_with(Path("datasets"), mock_hydra_config["dataset_configs"]["coco"])


@patch("src.datasets.dataset_manager.COCODownloader.download_and_extract")
def test_dataset_manager_extraction_failure(mock_coco_download, mock_hydra_config):
    """Test dataset manager extraction failure."""
    mock_coco_download.side_effect = ExtractionError("Failed to extract")
    manager = DatasetManager()

    # Verify results
    with pytest.raises(ExtractionError):
        manager.run_manager(mock_hydra_config["manager"], mock_hydra_config["dataset_configs"])

    mock_coco_download.assert_called_once_with(Path("datasets"), mock_hydra_config["dataset_configs"]["coco"])


@patch("src.datasets.dataset_manager.COCODownloader.download_and_extract")
def test_dataset_manager_invalid_input(mock_coco_download, mock_hydra_config):
    """Test dataset manager invalid input."""
    mock_coco_download.side_effect = InvalidInputError("Invalid URL")
    manager = DatasetManager()

    # Verify results
    with pytest.raises(InvalidInputError):
        manager.run_manager(mock_hydra_config["manager"], mock_hydra_config["dataset_configs"])

    mock_coco_download.assert_called_once_with(Path("datasets"), mock_hydra_config["dataset_configs"]["coco"])


def test_dataset_manager_invalid_type(mock_hydra_config):
    """Test dataset manager invalid type."""
    invalid_config = DictConfig({
        "manager": {
            "dataset_dir": "datasets",
            "types": ["invalid"]
        }
    })
    manager = DatasetManager()

    # Verify results
    with pytest.raises(ValueError):
        manager.run_manager(invalid_config, mock_hydra_config["dataset_configs"])
