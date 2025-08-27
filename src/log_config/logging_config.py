import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(
        name: str = "logger",
        log_dir: str = "logs",
        log_level: str = "INFO",
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 2
) -> logging.Logger:
    """
    Configure global logger.

    :param name: The name of the logger
    :param log_dir: Directory to store logs (ignored in test mode)
    :param log_level: Log level
    :param max_bytes: Max bytes to keep (ignored in test mode)
    :param backup_count: Number of backups to keep (ignored in test mode)
    :return: Logger instance
    """
    # Check if running in pytest
    is_test_mode = os.getenv("PYTEST_RUNNING", "0") == "1"

    # Create logger
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True if is_test_mode else False
    logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Log format
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S"
    )
    console_handler.setFormatter(log_format)

    # Add console handler
    logger.addHandler(console_handler)

    # Create file handler with size-based rotation, unless in test mode
    if not is_test_mode:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / f"{name}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger
