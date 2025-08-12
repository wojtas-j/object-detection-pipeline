from abc import ABC, abstractmethod
from pathlib import Path
from omegaconf import DictConfig

from src.exceptions.exceptions import InvalidInputError, DownloadError, ExtractionError
from src.datasets.datasets_utils import download_file, unzip_file
from src.log_config.logging_config import setup_logger

log = setup_logger(name=__name__)


class DatasetDownloader(ABC):
    """ Abstract class for downloading and extracting datasets methods. """

    @abstractmethod
    def get_config_path(self) -> str:
        """
        Return the path to the dataset YAML configuration file.

        :return: Path (str) to YAML configuration file (relative to configs/datasets/)
        """
        pass

    @abstractmethod
    def download(self, dataset_dir: str | Path, cfg: DictConfig) -> list[Path]:
        """
        Download dataset files to specified path.

        :param dataset_dir: Directory to download dataset files to
        :param cfg: Hydra configuration
        :return: List of paths to downloaded datasets
        :raises InvalidInputError: If the URL or destination path is invalid
        :raises DownloadError: If the download fails
        """
        pass

    @abstractmethod
    def extract(self, downloaded_files: list[Path], cfg: DictConfig, dataset_dir: str | Path) -> None:
        """
        Extract dataset files to specified path.

        :param downloaded_files: List of paths to downloaded datasets
        :param cfg: Hydra configuration
        :param dataset_dir: Directory for extraction
        :raises InvalidInputError: If the zip_path or extract_path is invalid
        :raises ExtractionError: If the extraction fails
        """
        pass

    def _download_files(self, dataset_dir: str | Path , cfg: DictConfig, dataset_name: str) -> list[Path]:
        """
        Download dataset files to specified path.

        :param dataset_dir: Directory to download dataset files to
        :param cfg: Hydra configuration
        :param dataset_name: Name of the dataset
        :return: List of paths to downloaded datasets
        :raises InvalidInputError: If the URL or destination path is invalid
        :raises DownloadError: If the download fails
        """
        downloaded_files = []
        if not hasattr(cfg.dataset, 'download_urls') or cfg.dataset.download_urls is None:
            log.info(f"No download URLs provided for {dataset_name}, skipping download")
            return downloaded_files
        for item in cfg.dataset.download_urls:
            url = item.url
            dest_name = item.file_name
            dest_path = Path(dataset_dir) / dest_name
            log.info(f"Downloading {dataset_name} dataset to {dest_path} from {url}")
            downloaded_file = download_file(url, dest_path, cfg)
            downloaded_files.append(downloaded_file)
        return downloaded_files

    def _extract_files(self, dataset_dir: str | Path, cfg: DictConfig, downloaded_files: list[Path], dataset_name: str) -> None:
        """
        Extract dataset files to specified path.

        :param dataset_dir: Directory to download dataset files to
        :param cfg: Hydra configuration
        :param downloaded_files: List of paths to downloaded datasets
        :param dataset_name: Name of the dataset
        :raises InvalidInputError: If the zip_path or extract_path is invalid
        :raises ExtractionError: If the extraction fails
        """
        extract_path = Path(dataset_dir) / cfg.dataset.extract_dir
        for file in downloaded_files:
            log.info(f"Extracting {dataset_name} dataset file {file} to {extract_path}")
            unzip_file(file, extract_path)

    def download_and_extract(self, dataset_dir: str | Path, cfg: DictConfig) -> None:
        """
        Download and extract dataset using Hydra configuration.

        :param dataset_dir: Directory to download dataset files to
        :param cfg: Hydra configuration
        :raises InvalidInputError: If the URL or destination path is invalid
        :raises ExtractionError: If the extraction fails
        :raises DownloadError: If the download fails
        """
        dataset_dir = Path(dataset_dir)

        try:
           downloaded_files = self.download(dataset_dir, cfg)
           self.extract(downloaded_files, cfg, dataset_dir)
        except (InvalidInputError, ExtractionError, DownloadError) as e:
            log.error(f"Failed to download and extract dataset: {e}")
            raise
        except Exception as e:
           log.error(f"Unexpected error during download and extract: {e}")
           raise


class COCODownloader(DatasetDownloader):
    """ Downloader for COCO dataset. """

    def get_config_path(self) -> str:
        """ Return path to COCO configuration file. """
        return "coco"

    def download(self, dataset_dir: str | Path, cfg: DictConfig) -> list[Path]:
        """ Download COCO dataset. """
        return self._download_files(dataset_dir, cfg, "coco")

    def extract(self, downloaded_files: list[Path], cfg: DictConfig, dataset_dir: str | Path) -> None:
        """ Extract COCO dataset. """
        self._extract_files(dataset_dir, cfg, downloaded_files, "coco")


class BDD100KDownloader(DatasetDownloader):
    """ Downloader for BDD100K dataset. """

    def get_config_path(self) -> str:
        """ Return path to BDD100K configuration file. """
        return "bdd100k"

    def download(self, dataset_dir: str | Path, cfg: DictConfig) -> list[Path]:
        """ Download BDD100K dataset. """
        return self._download_files(dataset_dir, cfg, "bdd100k")

    def extract(self, downloaded_files: list[Path], cfg: DictConfig, dataset_dir: str | Path) -> None:
        """ Extract BDD100K dataset. """
        self._extract_files(dataset_dir, cfg, downloaded_files, "bdd100k")
