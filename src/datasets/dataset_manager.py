import hydra
from pathlib import Path
from typing import TypeVar, Generic
from omegaconf import DictConfig

from src.exceptions.exceptions import InvalidInputError, DownloadError, ExtractionError
from src.datasets.dataset_downloader import DatasetDownloader, COCODownloader, BDD100KDownloader
from src.log_config.logging_config import setup_logger

log = setup_logger(__name__)
T = TypeVar('T', bound=DatasetDownloader)


class DatasetManager(Generic[T]):
    """ Generic manager for downloading and preparing multiple datasets. """

    def __init__(self, downloader: T | None = None):
        """
        Initialize with an optional downloader implementation.

        :param downloader: Instance of a dataset downloader
        """
        self.downloader = downloader

    def process(self, dataset_dir: str | Path, cfg: DictConfig) -> None:
        """
        Download, extract and prepare dataset.

        :param dataset_dir: Directory to store the dataset
        :param cfg: Hydra configuration
        :raises ValueError: If downloader is not set
        :raises InvalidInputError: If the URL or destination path is invalid
        :raises DownloadError: If the download fails
        :raises ExtractionError: If the extraction fails
        """
        if self.downloader is None:
            raise ValueError("Downloader not set")
        log.info(f"Processing dataset in {dataset_dir} with {self.downloader.__class__.__name__}")
        self.downloader.download_and_extract(dataset_dir, cfg)
        # TODO: Konwertery itd.

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
        downloader_map = {
            "coco": COCODownloader,
            "bdd100k": BDD100KDownloader,
        }

        for manager_type in manager_types:
            if manager_type not in downloader_map:
                raise ValueError(f"Unknown manager type: {manager_type}")
            self.downloader = downloader_map[manager_type]()
            try:
                self.process(dataset_dir, dataset_configs[manager_type])
            except (InvalidInputError, DownloadError, ExtractionError) as e:
                log.error(f"Failed to process dataset {manager_type}: {e}")
                raise
            except Exception as e:
                log.error(f"Unexpected error while processing dataset {manager_type}: {e}")
                raise


@hydra.main(config_path="../../configs", config_name="dataset_manager", version_base=None)
def main(cfg: DictConfig) -> None:
    dataset_configs = {}
    for dataset_type in cfg.manager.types:
        dataset_configs[dataset_type] = hydra.compose(config_name=dataset_type)

    manager = DatasetManager()
    manager.run_manager(cfg, dataset_configs)


if __name__ == "__main__":
    main()
