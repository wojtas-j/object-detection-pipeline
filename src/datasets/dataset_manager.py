import hydra
from pathlib import Path
from typing import TypeVar, Generic
from omegaconf import DictConfig

from src.datasets.dataset_converter import DatasetConverter, COCOConverter, BDD100KConverter
from src.exceptions.exceptions import InvalidInputError, DownloadError, ExtractionError
from src.datasets.dataset_downloader import DatasetDownloader, COCODownloader, BDD100KDownloader
from src.log_config.logging_config import setup_logger

log = setup_logger(__name__)
T = TypeVar('T', bound=DatasetDownloader)
U = TypeVar('U', bound=DatasetConverter)


class DatasetManager(Generic[T, U]):
    """ Generic manager for downloading and preparing multiple datasets. """

    def __init__(self, downloader: T | None = None, converter: U | None = None):
        """
        Initialize with an optional downloader implementation.

        :param downloader: Instance of a dataset downloader
        """
        self.downloader = downloader
        self.converter = converter

    def download(self, dataset_dir: str | Path, cfg: DictConfig) -> None:
        """
        Download the dataset and save it to the given path.

        :param dataset_dir: Directory to save the downloaded dataset.
        :param cfg: Hydra configuration.
        """
        if self.downloader is None:
            log.error("Downloader not set")
            raise ValueError("Downloader not set")
        log.info(f"Downloading dataset {cfg.dataset.name} to {dataset_dir}")
        self.downloader.download_and_extract(dataset_dir, cfg)

    def convert(self, dataset_dir: str | Path, cfg: DictConfig) -> None:
        """
        Convert the given dataset to the given format.

        :param dataset_dir: Directory to save the converted dataset.
        :param cfg: Hydra configuration.
        """
        if self.converter is None:
            log.error("Converter not set")
            raise ValueError("Converter not set")
        log.info(f"Converting dataset {cfg.dataset.name} to {dataset_dir}")
        self.converter.convert_and_process(dataset_dir, cfg)

    def execute_stage(self, dataset_dir: str | Path, cfg: DictConfig, stage: int) -> None:
        """
        Download, extract and prepare dataset.

        :param dataset_dir: Directory to store the dataset
        :param cfg: Hydra configuration
        :param stage: Manager stage
        :raises ValueError: If downloader is not set
        :raises InvalidInputError: If the URL or destination path is invalid
        :raises DownloadError: If the download fails
        :raises ExtractionError: If the extraction fails
        """
        dataset_dir = Path(dataset_dir)
        if stage == 0:
            self.download(dataset_dir, cfg)
        elif stage == 1:
            self.convert(dataset_dir, cfg)
        else:
            log.error(f"Invalid stage: {stage}")
            raise ValueError(f"Unknown stage: {stage}")

    def run_manager(self, cfg: DictConfig, dataset_configs: dict[str, DictConfig]):
        """
        Process multiple datasets from command line.

        :param cfg: Hydra configuration
        :param dataset_configs: Dataset configurations
        :raises ValueError: If downloader is not set
        :raises InvalidInputError: If the URL or destination path is invalid
        :raises DownloadError: If the download fails
        :raises ExtractionError: If the extraction fails
        """
        dataset_dir = Path(cfg.manager.dataset_dir)
        manager_types = cfg.manager.types
        stage = cfg.manager.stage
        downloader_map = {
            "coco": COCODownloader,
            "bdd100k": BDD100KDownloader,
        }
        converter_map = {
            "coco": COCOConverter,
            "bdd100k": BDD100KConverter,
        }

        for manager_type in manager_types:
            if stage == 0:
                if manager_type not in downloader_map:
                    log.error(f"Unknown downloader type: {manager_type}")
                    raise ValueError(f"Unknown downloader type: {manager_type}")
                self.downloader = downloader_map.get(manager_type)()
                self.converter = None
            elif stage == 1:
                if manager_type not in converter_map:
                    log.error(f"Unknown converter type: {manager_type}")
                    raise ValueError(f"Unknown converter type: {manager_type}")
                self.downloader = None
                self.converter = converter_map.get(manager_type)()
            try:
                self.execute_stage(dataset_dir, dataset_configs[manager_type], stage)
            except (InvalidInputError, DownloadError, ExtractionError, ValueError) as e:
                log.error(f"Failed to execute stage: {manager_type}: {e}")
                raise
            except Exception as e:
                log.error(f"Unexpected error while executing stage: {manager_type}: {e}")
                raise


@hydra.main(config_path="../../configs/datasets", config_name="dataset_manager", version_base=None)
def main(cfg: DictConfig) -> None:
    dataset_configs = {}
    for dataset_type in cfg.manager.types:
        dataset_configs[dataset_type] = hydra.compose(config_name=dataset_type)

    manager = DatasetManager()
    manager.run_manager(cfg, dataset_configs)


if __name__ == "__main__":
    main()
