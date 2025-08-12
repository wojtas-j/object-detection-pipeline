import pytest
import zipfile
import requests
from unittest.mock import patch, MagicMock
from omegaconf import DictConfig

from src.exceptions.exceptions import DownloadError, InvalidInputError, ExtractionError
from src.datasets.datasets_utils import path_exists, download_file, unzip_file


@pytest.fixture
def temp_dir(tmp_path):
    """ Create a temporary directory. """
    return tmp_path


@pytest.fixture
def sample_zip(temp_dir):
    """ Create a sample ZIP file. """
    zip_path = temp_dir / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zip_file:
        zip_file.writestr("file1.txt", " Content file1")
        zip_file.writestr("file2.txt", " Content file2")

    return zip_path


@pytest.fixture
def mock_config():
    """ Mock Hydra configuration. """
    return DictConfig({
        "dataset": {
            "download_timeout": 60,
            "chunk_size": 8192
        }
    })


def test_path_exists(temp_dir):
    """ Test path_exists function for existing and non-existing paths. """
    # Verify results
    assert path_exists(temp_dir) is True, f"Expected existing path '{temp_dir}' to return True"
    assert path_exists(temp_dir / "nonexsting") is False, f"Expected non-existing path to return False"


def test_download_file_invalid_url(mock_config):
    """ Test download_file with invalid url. """
    # Verify results
    with pytest.raises(InvalidInputError):
        download_file("invalid_url", "test.zip", mock_config)
    with pytest.raises(InvalidInputError):
        download_file("", "test.zip", mock_config)


def test_download_file_no_extension(temp_dir, mock_config):
    """ Test download_file with destination path lacking extension. """
    # Verify results
    with pytest.raises(InvalidInputError):
        download_file("http://example.com/file.zip", temp_dir / "test", mock_config)


@patch("src.datasets.datasets_utils.requests.get")
def test_download_file_success(mock_get, temp_dir, mock_config):
    """ Test download_file with success download. """
    # Mock HTTP response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-length": 1024}
    mock_response.iter_content.return_value = [b"data" * 256]
    mock_response.raise_for_status.return_value = None

    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = None
    mock_get.return_value = mock_response

    # Test download
    dest_path = temp_dir / "test.zip"
    result = download_file("http://example.com/file.zip", dest_path, mock_config)

    # Verify results
    assert result == dest_path, f"Expected download_file to return '{dest_path}', got '{result}'"
    assert dest_path.exists(), f"Expected downloaded file '{dest_path}' to exist"
    assert dest_path.stat().st_size == 1024, f"Expected downloaded file size 1024, got {dest_path.stat().st_size}"
    mock_get.assert_called_once_with("http://example.com/file.zip", stream=True, timeout=60)


@patch("src.datasets.datasets_utils.requests.get")
def test_download_file_size_mismatch(mock_get, temp_dir,mock_config):
    """ Test download_file with size mismatch. """
    # Mock HTTP response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-length": 1024}
    mock_response.iter_content.return_value = [b"data" * 256]
    mock_response.raise_for_status.return_value = None

    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = None
    mock_get.return_value = mock_response

    # Test download
    dest_path = temp_dir / "test.zip"
    result = download_file("http://example.com/file.zip", dest_path, mock_config)

    # Verify results
    assert result == dest_path, f"Expected download_file to return '{dest_path}', got '{result}'"
    assert dest_path.exists(), f"Expected downloaded file '{dest_path}' to exist"
    assert dest_path.stat().st_size == 1024, f"Expected downloaded file size 1024, got {dest_path.stat().st_size}"


@patch("src.datasets.datasets_utils.requests.get")
def test_download_file_failure(mock_get, temp_dir, mock_config):
    """ Test download_file with failure download. """
    # Mock failed HTTP response
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.RequestException("Connection error")

    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = None
    mock_get.return_value = mock_response

    # Verify results
    dest_path = temp_dir / "test.zip"
    with pytest.raises(DownloadError):
        download_file("http://example.com/file.zip", dest_path, mock_config)
    assert not dest_path.exists(), f"File '{dest_path}' should not exist after failed download"


def test_unzip_file_success(temp_dir, sample_zip):
    """ Test unzip_file with a valid ZIP file. """
    extract_dir = temp_dir / "extracted"
    unzip_file(sample_zip, extract_dir)

    # Verify results
    assert extract_dir.exists(), f"Expected extraction directory '{extract_dir}' to exist"
    assert (extract_dir / "file1.txt").exists(), "Expected 'file1.txt' to be extracted"
    assert (extract_dir / "file2.txt").exists(), "Expected 'file2.txt' to be extracted"
    assert len(list(extract_dir.rglob("*"))) == 2, f"Expected 2 files after extraction, got {len(list(extract_dir.rglob('*')))}"


def test_unzip_file_nonexistent(temp_dir):
    """ Test unzip_file with a nonexistent ZIP file. """

    # Verify results
    with pytest.raises(InvalidInputError):
        unzip_file(temp_dir / "nonexistent", temp_dir / "extracted")


def test_unzip_file_invalid_extension(temp_dir):
    """ Test unzip_file with an invalid extension. """
    invalid_file = temp_dir / "test.txt"
    invalid_file.touch()

    # Verify results
    with pytest.raises(InvalidInputError):
        unzip_file(invalid_file, temp_dir / "extracted")


def test_unzip_file_not_directory(temp_dir, sample_zip):
    """ Test unzip_file with a non-directory ZIP file. """
    invalid_extract = temp_dir / "not_a_dir.txt"
    invalid_extract.touch()

    # Verify results
    with pytest.raises(InvalidInputError):
        unzip_file(sample_zip, invalid_extract)


def test_unzip_file_empty_zip(temp_dir):
    """ Test unzip_file with an empty ZIP file. """
    # Create empty zip
    empty_zip = temp_dir / "empty.zip"
    with zipfile.ZipFile(empty_zip, "w"):
        pass

    extract_dir = temp_dir / "extracted"
    unzip_file(empty_zip, extract_dir)

    # Verify results
    assert extract_dir.exists(), f"Expected extraction directory '{extract_dir}' to exist"
    assert len(list(extract_dir.rglob('*.txt'))) == 0, f"Expected 0 .txt files in extracted empty zip, got {len(list(extract_dir.rglob('*.txt')))}"


@patch("src.datasets.datasets_utils.zipfile.ZipFile")
def test_unzip_file_permission_error(mock_zipfile, temp_dir, sample_zip):
    """ Test unzip_file with an invalid permission error. """
    mock_zipfile.side_effect = PermissionError("Permission denied")

    # Verify results
    with pytest.raises(ExtractionError):
        unzip_file(sample_zip, temp_dir / "extracted")
