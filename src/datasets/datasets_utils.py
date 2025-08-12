import time
import zipfile
import requests
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm
from omegaconf import DictConfig

from src.exceptions.exceptions import InvalidInputError, DownloadError, ExtractionError
from src.log_config.logging_config import setup_logger

log = setup_logger(name = __name__)


def path_exists(path: str | Path) -> bool:
    """
    Check if a path exists.

    :param path: Path to check (string or Path object)
    :return: True if path exists, False otherwise
    """
    path = Path(path)
    log.debug(f"Check if {path} exists")

    return path.exists()


def download_file(url: str, dest_path: str | Path, cfg: DictConfig) -> Path:
    """
    Download a file from a URL.

    :param url: URL to download
    :param dest_path: Path to download to
    :param cfg: Hydra configuration
    :return: Path to downloaded file
    :raises InvalidInputError: If the URL or destination path is invalid
    :raises DownloadError: If the download fails
    """
    timeout = cfg.dataset.download_timeout
    chunk_size = cfg.dataset.chunk_size
    dest_path = Path(dest_path)

    # Input validation
    if not url or not isinstance(url, str) or not urlparse(url).scheme:
        log.error(f"Invalid URL: {url}")
        raise InvalidInputError(f"Invalid URL: {url}")

    if not dest_path.suffix:
        log.error(f"Destination path must end with an extension: {dest_path}")
        raise InvalidInputError(f"Destination path must end with an extension: {dest_path}")

    # Check if the directory exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    log.debug(f"Check if {dest_path} exists")

    # Check if file already exists and matches expected size
    if path_exists(dest_path):
        try:
            with requests.head(url, timeout=timeout) as r:
                r.raise_for_status()
                expected_size = int(r.headers.get("content-length", 0))
            actual_size = dest_path.stat().st_size
            if expected_size == 0 or actual_size == expected_size:
                log.info(f"File {dest_path} already exists and matches expected size ({actual_size} bytes), skipping download")
                return dest_path
            else:
                log.warning(f"File {dest_path} exists but size ({actual_size} bytes) does not match expected ({expected_size} bytes), re-downloading")
                dest_path.unlink()
        except requests.exceptions.RequestException as e:
            log.warning(f"Could not verify file size for {url} ({e}), proceeding with download")

    # Download file
    try:
        start_time = time.time()
        log.info(f"Start downloading {url} to {dest_path}")
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            log.debug(f"Excepted file size: {total_size}")
            with dest_path.open("wb") as f, tqdm(
                    total=total_size, unit="B", unit_scale=True, desc="Downloading"
            ) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            download_time = time.time() - start_time
            actual_size = dest_path.stat().st_size
            log.info(f"Downloaded file to: {dest_path}, size: {actual_size} bytes, time: {download_time:.2f} seconds")

            if 0 < total_size != actual_size:
                log.warning(f"Downloaded file ({actual_size} bytes) does not match expected file size ({total_size} bytes)")

    except requests.exceptions.RequestException as e:
        log.error(f"Failed to download {url}: {e}")
        dest_path.unlink(missing_ok=True)
        raise DownloadError(f"Failed to download {url}: {e}")

    return dest_path


def unzip_file(zip_path: str | Path, extract_path: str | Path) -> None:
    """
    Extract a file from a zip file to a specified path.

    :param zip_path: Path to the zip file
    :param extract_path: Directory to extract zip file to
    :raises InvalidInputError: If the zip_path or extract_path is invalid
    :raises ExtractionError: If the extraction fails
    """

    zip_path = Path(zip_path)
    extract_path = Path(extract_path).resolve()

    # Input validation
    if not zip_path.exists():
        log.error(f"Zip file {zip_path} does not exist")
        raise InvalidInputError(f"Zip file {zip_path} does not exist")

    if not zip_path.suffix.lower() == ".zip":
        log.error(f"File {zip_path} is not a zip file")
        raise InvalidInputError(f"File {zip_path} is not a zip file")

    if extract_path.exists() and not extract_path.is_dir():
        log.error(f"Extract path {extract_path} is not a directory")
        raise InvalidInputError(f"Extract path {extract_path} is not a directory")

    # Check if the directory exists
    extract_path.mkdir(parents=True, exist_ok=True)
    log.debug(f"Extracting {zip_path} to {extract_path}")

    # Extract file
    try:
        log.info(f"Extracting {zip_path} to {extract_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
            extracted_files = len(list(extract_path.rglob("*")))
            log.info(f"Extracted {extracted_files} files")
    except zipfile.BadZipFile as e:
        log.error(f"Failed to extract {zip_path}, invalid ZIP file: {e}")
        raise ExtractionError(f"Failed to extract {zip_path}, invalid ZIP file: {e}")
    except PermissionError as e:
        log.error(f"Failed to extract {zip_path}, permission denied: {e}")
        raise ExtractionError(f"Failed to extract {zip_path}, permission denied: {e}")
    except Exception as e:
        log.error(f"Failed to extract {zip_path}: {e}")
        raise ExtractionError(f"Failed to extract {zip_path}: {e}")
