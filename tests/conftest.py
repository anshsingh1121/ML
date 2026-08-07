"""
Shared pytest fixtures and configuration for unit and integration tests.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from src.utils.config_manager import ConfigManager
from src.utils.logger import LoggerFactory


@pytest.fixture(autouse=True)
def reset_singletons() -> Generator[None, None, None]:
    """Ensure clean Singleton state before and after each test."""
    ConfigManager.reset()
    LoggerFactory.reset()
    yield
    ConfigManager.reset()
    LoggerFactory.reset()


@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Create an isolated temporary workspace directory with sample configs."""
    temp_dir = tempfile.mkdtemp(prefix="fcb_iip_test_")
    workspace = Path(temp_dir)

    # Create config folder
    config_dir = workspace / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Create sample config.yaml
    sample_config = """
app:
  name: "Test Incident Intelligence"
  version: "1.0.0-test"
  environment: "test"

data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  default_format: "csv"

dataset_generator:
  seed: 999
  output_filename: "test_incidents"
  category_distribution:
    Network: 0.5
    Software: 0.5
  priority_distribution:
    1: 0.2
    2: 0.8
"""
    with open(config_dir / "config.yaml", "w", encoding="utf-8") as f:
        f.write(sample_config)

    # Create sample logging.yaml
    sample_logging = """
version: 1
disable_existing_loggers: false
formatters:
  simple:
    format: "%(levelname)s | %(message)s"
handlers:
  console:
    class: "logging.StreamHandler"
    level: "DEBUG"
    formatter: "simple"
root:
  level: "DEBUG"
  handlers: ["console"]
"""
    with open(config_dir / "logging.yaml", "w", encoding="utf-8") as f:
        f.write(sample_logging)

    # Change CWD temporarily or pass config_dir explicitly
    orig_cwd = os.getcwd()
    os.chdir(workspace)

    yield workspace

    os.chdir(orig_cwd)
    shutil.rmtree(temp_dir, ignore_errors=True)
