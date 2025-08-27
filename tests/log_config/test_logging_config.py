import pytest
import logging
from unittest.mock import patch

from src.log_config.logging_config import setup_logger


#region Fixtures
@pytest.fixture
def log_dir(tmp_path):
    """ Create a temporary directory for logs (used in non-test mode). """
    return tmp_path / "logs"
#endregion

# Logging_config Tests
def test_logger_levels_test_mode(log_dir, caplog):
    """ Test logger levels in test mode (console only, no file). """
    caplog.set_level(logging.DEBUG)
    logger = setup_logger(
        name="test_logger",
        log_dir=str(log_dir),
        log_level="DEBUG"
    )

    # Logs at different levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    # Verify results
    # Check that no log file or directory is created
    log_file = log_dir / "test_logger.log"
    assert not log_dir.exists(), "Log directory should not be created in test mode"
    assert not log_file.exists(), "Log file should not exist in test mode"

    # Verify console logs
    assert len(caplog.records) == 5, "Expected 5 log records"
    messages = [record.message for record in caplog.records]
    assert "Debug message" in messages, "Console log does not contain Debug message"
    assert "Info message" in messages, "Console log does not contain Info message"
    assert "Warning message" in messages, "Console log does not contain Warning message"
    assert "Error message" in messages, "Console log does not contain Error message"
    assert "Critical message" in messages, "Console log does not contain Critical message"


def test_logger_invalid_level_test_mode(log_dir, caplog):
    """ Test logger uses default level with invalid log_level in test mode (console only). """
    caplog.set_level(logging.DEBUG)
    logger = setup_logger(
        name="test_invalid",
        log_dir=str(log_dir),
        log_level="INVALID"
    )

    # Log at different levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    # Verify results
    # Check that no log file or directory is created
    log_file = log_dir / "test_invalid.log"
    assert not log_dir.exists(), "Log directory should not be created in test mode"
    assert not log_file.exists(), "Log file should not exist in test mode"

    # Verify console logs
    assert len(caplog.records) == 4, "Expected 4 log records (DEBUG should be skipped)"
    messages = [record.message for record in caplog.records]
    assert "Debug message" not in messages, "Console log should not contain Debug message"
    assert "Info message" in messages, "Console log does not contain Info message"
    assert "Warning message" in messages, "Console log does not contain Warning message"
    assert "Error message" in messages, "Console log does not contain Error message"
    assert "Critical message" in messages, "Console log does not contain Critical message"


def test_logger_no_file_creation_test_mode(log_dir, caplog):
    """ Test that logger does not create log directory or file in test mode. """
    caplog.set_level(logging.INFO)
    logger = setup_logger(
        name="test_rotation",
        log_dir=str(log_dir),
        log_level="INFO",
        max_bytes=1000,
        backup_count=1
    )

    # Log messages to trigger potential file creation
    message = "*" * 250
    for _ in range(5):
        logger.info(message)

    # Verify results
    # Check that no log directory or file is created
    assert not log_dir.exists(), "Log directory should not be created in test mode"
    log_file = log_dir / "test_rotation.log"
    assert not log_file.exists(), "Log file should not be created in test mode"

    # Verify console logs
    assert len(caplog.records) == 5, "Expected 5 log records"
    messages = [record.message for record in caplog.records]
    assert all(message in messages for message in messages), "Console log does not contain expected messages"


@patch.dict("os.environ", {"PYTEST_RUNNING": "0"})
def test_logger_levels_non_test_mode(log_dir):
    """ Test logger levels in non-test mode (logs to file and console). """
    logger = setup_logger(
        name="test_logger",
        log_dir=str(log_dir),
        log_level="DEBUG"
    )

    # Logs at different levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    # Verify results
    # Check if log file exists
    log_file = log_dir / "test_logger.log"
    assert log_file.exists(), "Log file does not exist in non-test mode"

    # Read and verify log content
    with log_file.open("r") as f:
        content = f.read()
        assert "Debug message" in content, "Log file does not contain Debug message"
        assert "Info message" in content, "Log file does not contain Info message"
        assert "Warning message" in content, "Log file does not contain Warning message"
        assert "Error message" in content, "Log file does not contain Error message"
        assert "Critical message" in content, "Log file does not contain Critical message"


@patch.dict("os.environ", {"PYTEST_RUNNING": "0"})
def test_logger_invalid_level_non_test_mode(log_dir):
    """ Test logger uses default level with invalid log_level in non-test mode (logs to file). """
    logger = setup_logger(
        name="test_invalid",
        log_dir=str(log_dir),
        log_level="INVALID"
    )

    # Log at different levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    # Verify results
    # Check if log file exists
    log_file = log_dir / "test_invalid.log"
    assert log_file.exists(), "Log file does not exist in non-test mode"

    # Read and verify log content
    with log_file.open("r") as f:
        content = f.read()
        assert "Debug message" not in content, "Log file should not contain Debug message"
        assert "Info message" in content, "Log file does not contain Info message"
        assert "Warning message" in content, "Log file does not contain Warning message"
        assert "Error message" in content, "Log file does not contain Error message"
        assert "Critical message" in content, "Log file does not contain Critical message"

@patch.dict("os.environ", {"PYTEST_RUNNING": "0"})
def test_logger_rotation_non_test_mode(log_dir):
    """ Test logger rotates log files when size limit is exceeded in non-test mode. """
    logger = setup_logger(
        name="test_rotation",
        log_dir=str(log_dir),
        log_level="INFO",
        max_bytes=1000,
        backup_count=1
    )

    # Create and write data to trigger rotation
    message = "*" * 250
    for _ in range(5):
        logger.info(message)

    # Verify results
    # Check if log files exist
    log_file = log_dir / "test_rotation.log"
    rotated_file = log_dir / "test_rotation.log.1"
    assert log_file.exists(), "Log file was not created in non-test mode"
    assert rotated_file.exists(), "Rotated log file was not created in non-test mode"

    # Read and verify content in rotated file
    with rotated_file.open("r") as f:
        content = f.read()
        assert message in content, "Rotated log file should contain message"
#endregion