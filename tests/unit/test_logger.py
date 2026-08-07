"""Unit tests for LoggerFactory and custom log formatting."""

import logging
from pathlib import Path
import sys

import pytest

from src.utils.logger import LoggerFactory, get_logger, JsonFormatter


def test_logger_factory_singleton_and_configure(temp_workspace: Path) -> None:
    """Verify LoggerFactory initializes, configures from YAML, and retrieves loggers correctly."""
    LoggerFactory.reset()
    config_yaml = temp_workspace / "config" / "logging.yaml"
    LoggerFactory.configure(config_path=str(config_yaml), log_dir=str(temp_workspace / "logs"))

    logger1 = get_logger("test_module_1")
    logger2 = get_logger("test_module_2")
    assert isinstance(logger1, logging.Logger)
    assert isinstance(logger2, logging.Logger)
    assert logger1.name == "test_module_1"
    assert logger2.name == "test_module_2"


def test_logger_json_formatter_with_exception_and_extras() -> None:
    """Verify JSON formatting behavior when configured with exceptions and extra fields."""
    formatter = JsonFormatter()

    try:
        raise ValueError("Simulated error for logger testing")
    except ValueError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test_json_logger",
        level=logging.ERROR,
        pathname="test.py",
        lineno=42,
        msg="Error occurred: %s",
        args=("critical_failure",),
        exc_info=exc_info
    )
    # Inject extra field
    record.__dict__["incident_id"] = "INC0012345"
    record.__dict__["latency_ms"] = 145.2

    formatted = formatter.format(record)
    assert "Error occurred: critical_failure" in formatted
    assert "ERROR" in formatted
    assert "Simulated error for logger testing" in formatted
    assert "INC0012345" in formatted
    assert "145.2" in formatted


def test_logger_fallback_and_invalid_yaml(temp_workspace: Path) -> None:
    """Verify LoggerFactory falls back to basic configuration if YAML config is missing or invalid."""
    LoggerFactory.reset()
    bad_config = temp_workspace / "config" / "bad_logging.yaml"
    with open(bad_config, "w", encoding="utf-8") as f:
        f.write("invalid: [yaml: syntax:")

    LoggerFactory.configure(config_path=str(bad_config), log_dir=str(temp_workspace / "fallback_logs"))
    logger = get_logger("fallback_test")
    logger.info("Fallback log message worked")
    assert Path(temp_workspace / "fallback_logs" / "app.log").exists()
