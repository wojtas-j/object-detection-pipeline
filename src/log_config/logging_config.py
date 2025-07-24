import logging
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
    Configure of global logger.

    :param name: The name of the logger
    :param log_dir: Directory to store logs
    :param log_level: Log level
    :param max_bytes: Max bytes to keep
    :param backup_count: Number of backups to keep
    :return: Logger instance
    """

    # Create directory if not exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create logger
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    # Create file handler with size based rotation
    if logger.handlers:
        return logger

    log_file = log_path / f"{name}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Log format
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S"
    )
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
