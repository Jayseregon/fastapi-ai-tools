import logging
from unittest.mock import patch

import pytest
from rich.logging import RichHandler

from src.configs.log_config import configure_logging


@pytest.fixture
def reset_logging():
    """Reset logging configuration after each test"""
    yield
    logging.getLogger().handlers.clear()


def test_configure_logging_basic_setup(reset_logging):
    """Test basic logging configuration setup"""
    configure_logging()
    logger = logging.getLogger("src")

    assert logger.level == logging.INFO  # Default level for non-dev
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], RichHandler)


def test_configure_logging_formatters(reset_logging):
    """Test logging formatters configuration"""
    configure_logging()
    logger = logging.getLogger("src")
    handler = logger.handlers[0]

    # Check formatter pattern matches expected format
    expected_format = "(%(correlation_id)s) %(name)s:%(lineno)d - %(message)s"
    assert handler.formatter._fmt == expected_format


@patch("src.configs.log_config.config")
@patch("src.configs.log_config.isinstance")
def test_configure_logging_dev_environment(mock_isinstance, mock_config, reset_logging):
    """Test logging configuration in dev environment"""
    mock_isinstance.return_value = True

    configure_logging()
    logger = logging.getLogger("src")

    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], RichHandler)


def test_configure_logging_uvicorn_logger(reset_logging):
    """Test uvicorn logger configuration"""
    configure_logging()
    uvicorn_logger = logging.getLogger("uvicorn")

    assert uvicorn_logger.level == logging.INFO
    assert len(uvicorn_logger.handlers) == 1


def test_configure_logging_database_loggers(reset_logging):
    """Test database loggers configuration"""
    configure_logging()

    db_logger = logging.getLogger("databases")
    aiosqlite_logger = logging.getLogger("aiosqlite")

    assert db_logger.level == logging.WARNING
    assert aiosqlite_logger.level == logging.WARNING


def test_configure_logging_correlation_id_filter(reset_logging):
    """Test correlation ID filter configuration"""
    configure_logging()
    logger = logging.getLogger("src")

    # Check if correlation ID filter is applied
    handler = logger.handlers[0]
    assert any(hasattr(f, "uuid_length") for f in handler.filters)


@patch("src.configs.log_config.isinstance")
def test_correlation_id_length(mock_isinstance, reset_logging):
    """Test correlation ID length in different environments"""
    # Test dev environment
    mock_isinstance.return_value = True
    configure_logging()
    logger = logging.getLogger("src")
    filter_dev = next(
        f for f in logger.handlers[0].filters if hasattr(f, "uuid_length")
    )
    assert filter_dev.uuid_length == 8

    # Test non-dev environment
    mock_isinstance.return_value = False
    configure_logging()
    logger = logging.getLogger("src")
    filter_prod = next(
        f for f in logger.handlers[0].filters if hasattr(f, "uuid_length")
    )
    assert filter_prod.uuid_length == 32
