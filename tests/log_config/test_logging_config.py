import pytest

from src.log_config.logging_config import setup_logger


@pytest.fixture
def log_dir(tmp_path):
    """ Create a temporary directory. """
    return tmp_path / "logs"


def test_logger_levels(log_dir):
    """ Test log_config levels. """
    logger = setup_logger(
        name = "test_logger",
        log_dir = str(log_dir),
        log_level = "DEBUG"
    )

    # Logs at different levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    # Chceck if logs file exists
    log_file = log_dir / "test_logger.log"
    assert log_file.exists(), "Log file does not exist"

    # Read and verify log content
    with log_file.open("r") as f:
        content = f.read()
        assert "Debug message" in content, "Log file does not contain Debug message"
        assert "Info message" in content, "Log file does not contain Info message"
        assert "Warning message" in content, "Log file does not contain Warning message"
        assert "Error message" in content, "Log file does not contain Error message"
        assert "Critical message" in content, "Log file does not contain Critical message"


def test_logger_invalid_level(log_dir):
    """ Test that log_config uses default level with invalid log_level. """
    logger = setup_logger(
        name = "test_invalid",
        log_dir = str(log_dir),
        log_level = "INVALID"
    )

    # Log at different levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    # Chceck if logs file exists
    log_file = log_dir / "test_invalid.log"
    assert log_file.exists(), "Log file does not exist"

    # Read and verify log content
    with log_file.open("r") as f:
        content = f.read()
        assert "Debug message" not in content, "Log file should not contain debug message"
        assert "Info message" in content, "Log file does not contain Info message"
        assert "Warning message" in content, "Log file does not contain Warning message"
        assert "Error message" in content, "Log file does not contain Error message"
        assert "Critical message" in content, "Log file does not contain Critical message"


def test_logger_rotation(log_dir):
    """ Test that log_config rotates log files when size limit is exceeded. """
    logger = setup_logger(
        name = "test_rotation",
        log_dir = str(log_dir),
        log_level = "INFO",
        max_bytes = 1000,
        backup_count = 1
    )

    # Create and write data to trigger rotation
    message = "*" * 250
    for _ in range(5):
        logger.info(message)

    # Check if logs file exists
    log_file = log_dir / "test_rotation.log"
    rotated_file = log_dir / "test_rotation.log.1"
    assert log_file.exists(), "Log file was not created"
    assert rotated_file.exists(), "Rotated log file was not created"

    # Read and verify content in rotated file
    with rotated_file.open("r") as f:
        content = f.read()
        assert message in content, "Rotated log file should contain message"