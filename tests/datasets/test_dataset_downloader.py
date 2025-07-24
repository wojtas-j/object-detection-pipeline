import pytest
from unittest.mock import patch
from omegaconf import DictConfig

from src.exceptions.exceptions import DownloadError, ExtractionError, InvalidInputError
from src.datasets.dataset_downloader import COCODownloader, BDD100KDownloader


@pytest.fixture
def temp_dir(tmp_path):
    """ Create a temporary directory. """
    return tmp_path


@pytest.fixture
def mock_hydra_config():
    """ Mock Hydra configuration. """
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
    return {"coco": coco_config, "bdd100k": bdd100k_config}


@patch("src.datasets.dataset_downloader.download_file")
@patch("src.datasets.dataset_downloader.unzip_file")
def test_coco_downloader_success(mock_unzip, mock_download, mock_hydra_config, temp_dir):
    """ Test COCODownloader success case. """
    downloader = COCODownloader()
    mock_download.side_effect = [
        temp_dir / "train2017.zip",
        temp_dir / "val2017.zip",
        temp_dir / "annotations_trainval2017.zip"
    ]

    downloader.download_and_extract(temp_dir, mock_hydra_config["coco"])

    # Verify results
    assert mock_download.call_count == 3
    mock_download.assert_any_call(
        "http://images.cocodataset.org/zips/train2017.zip",
        temp_dir / "train2017.zip",
        mock_hydra_config["coco"]
    )
    assert mock_unzip.call_count == 3
    mock_unzip.assert_any_call(
        temp_dir / "train2017.zip",
        temp_dir / "coco"
    )


@patch("src.datasets.dataset_downloader.download_file")
@patch("src.datasets.dataset_downloader.unzip_file")
def test_coco_downloader_download_failure(mock_unzip, mock_download, mock_hydra_config, temp_dir):
    """ Test COCODownloader failure case. """
    downloader = COCODownloader()
    mock_download.side_effect = DownloadError("Failed to download")

    # Verify results
    with pytest.raises(DownloadError):
        downloader.download_and_extract(temp_dir, mock_hydra_config["coco"])

    assert mock_download.called
    assert not mock_unzip.called

@patch("src.datasets.dataset_downloader.download_file")
@patch("src.datasets.dataset_downloader.unzip_file")
def test_coco_downloader_extract_failure(mock_unzip, mock_download, mock_hydra_config, temp_dir):
    """ Test COCODownloader extraction failure case. """
    downloader = COCODownloader()
    mock_download.side_effect = [
        temp_dir / "train2017.zip",
        temp_dir / "val2017.zip",
        temp_dir / "annotations_trainval2017.zip"
    ]

    mock_unzip.side_effect = ExtractionError("Failed to extract")

    # Verify results
    with pytest.raises(ExtractionError):
        downloader.download_and_extract(temp_dir, mock_hydra_config["coco"])

    assert mock_download.call_count == 3
    assert mock_unzip.called

@patch("src.datasets.dataset_downloader.download_file")
@patch("src.datasets.dataset_downloader.unzip_file")
def test_coco_downloader_invalid_input(mock_unzip, mock_download, mock_hydra_config, temp_dir):
    """ Test COCODownloader invalid input. """
    downloader = COCODownloader()
    mock_download.side_effect = InvalidInputError("Invalid URL")

    # Verify results
    with pytest.raises(InvalidInputError):
        downloader.download_and_extract(temp_dir, mock_hydra_config["coco"])

    assert mock_download.called
    assert not mock_unzip.called


@patch("src.datasets.dataset_downloader.download_file")
@patch("src.datasets.dataset_downloader.unzip_file")
def test_bdd100k_downloader_success(mock_unzip, mock_download, mock_hydra_config, temp_dir):
    """ Test BDD100KDownloader success cases. """
    downloader = BDD100KDownloader()
    mock_download.side_effect = [
        temp_dir / "bdd100k_videos_train_00.zip",
        temp_dir / "bdd100k_videos_val_00.zip",
        temp_dir / "bdd100k_labels.zip"
    ]

    downloader.download_and_extract(temp_dir, mock_hydra_config["bdd100k"])

    # Verify results
    assert mock_download.call_count == 3
    mock_download.assert_any_call(
        "http://dl.yf.io/bdd100k/video_parts/bdd100k_videos_train_00.zip",
        temp_dir / "bdd100k_videos_train_00.zip",
        mock_hydra_config["bdd100k"]
    )
    assert mock_unzip.call_count == 3
    mock_unzip.assert_any_call(
        temp_dir / "bdd100k_videos_train_00.zip",
        temp_dir / "bdd100k"
    )


@patch("src.datasets.dataset_downloader.download_file")
@patch("src.datasets.dataset_downloader.unzip_file")
def test_bdd100k_downloader_download_failure(mock_unzip, mock_download, mock_hydra_config, temp_dir):
    """ Test BDD100KDownloader download failure case. """
    downloader = BDD100KDownloader()
    mock_download.side_effect = DownloadError("Failed to download")

    # Verify results
    with pytest.raises(DownloadError):
        downloader.download_and_extract(temp_dir, mock_hydra_config["bdd100k"])

    assert mock_download.called
    assert not mock_unzip.called


@patch("src.datasets.dataset_downloader.download_file")
@patch("src.datasets.dataset_downloader.unzip_file")
def test_bdd100k_downloader_extract_failure(mock_unzip, mock_download, mock_hydra_config, temp_dir):
    """ Test BDD100KDownloader extract failure case. """
    downloader = BDD100KDownloader()
    mock_download.side_effect = [
        temp_dir / "bdd100k_videos_train_00.zip",
        temp_dir / "bdd100k_videos_val_00.zip",
        temp_dir / "bdd100k_labels.zip"
    ]
    mock_unzip.side_effect = ExtractionError("Failed to extract")

    # Verify results
    with pytest.raises(ExtractionError):
        downloader.download_and_extract(temp_dir, mock_hydra_config["bdd100k"])

    assert mock_download.call_count == 3
    assert mock_unzip.called


@patch("src.datasets.dataset_downloader.download_file")
@patch("src.datasets.dataset_downloader.unzip_file")
def test_bdd100k_downloader_invalid_input(mock_unzip, mock_download, mock_hydra_config, temp_dir):
    """ Test BDD100KDownloader invalid input. """
    downloader = BDD100KDownloader()
    mock_download.side_effect = InvalidInputError("Invalid URL")

    # Verify results
    with pytest.raises(InvalidInputError):
        downloader.download_and_extract(temp_dir, mock_hydra_config["bdd100k"])

    assert mock_download.called
    assert not mock_unzip.called
