"""Unit tests for ConfigManager."""

import os
from pathlib import Path

import pytest

from src.utils.config_manager import ConfigManager


def test_config_manager_singleton(temp_workspace: Path) -> None:
    """Verify ConfigManager behaves as a thread-safe Singleton."""
    cfg1 = ConfigManager(config_dir=str(temp_workspace / "config"))
    cfg2 = ConfigManager(config_dir=str(temp_workspace / "config"))
    assert cfg1 is cfg2


def test_config_get_dot_notation(temp_workspace: Path) -> None:
    """Verify dot-notation retrieval of nested keys."""
    cfg = ConfigManager(config_dir=str(temp_workspace / "config"))
    assert cfg.get("app.name") == "Test Incident Intelligence"
    assert cfg.get("data.raw_dir") == "data/raw"
    assert cfg.get("non_existent.key", "default_val") == "default_val"


def test_config_get_section(temp_workspace: Path) -> None:
    """Verify retrieving full dictionary sections."""
    cfg = ConfigManager(config_dir=str(temp_workspace / "config"))
    data_section = cfg.get_section("data")
    assert isinstance(data_section, dict)
    assert data_section.get("default_format") == "csv"
    assert cfg.get_section("non_existent_section") == {}


def test_config_env_var_interpolation(temp_workspace: Path) -> None:
    """Verify environment variable substitution (${VAR:default}) in strings and lists."""
    os.environ["TEST_ENV_VAR"] = "production_override"
    config_dir = temp_workspace / "config"
    with open(config_dir / "config.yaml", "a", encoding="utf-8") as f:
        f.write("\napp:\n  custom_setting: \"${TEST_ENV_VAR:fallback}\"\n  missing_setting: \"${NON_EXISTENT_VAR:my_fallback}\"\n  list_setting: [\"${TEST_ENV_VAR:val}\", \"normal_str\"]\n")

    ConfigManager.reset()
    cfg = ConfigManager(config_dir=str(config_dir))
    assert cfg.get("app.custom_setting") == "production_override"
    assert cfg.get("app.missing_setting") == "my_fallback"
    assert cfg.get("app.list_setting") == ["production_override", "normal_str"]


def test_config_has_all_reload_repr(temp_workspace: Path) -> None:
    """Verify has(), all property, reload(), and __repr__."""
    cfg = ConfigManager(config_dir=str(temp_workspace / "config"))
    assert cfg.has("app.name") is True
    assert cfg.has("non_existent.path") is False
    assert isinstance(cfg.all, dict)
    assert "ConfigManager" in repr(cfg)

    # Test reload
    cfg.reload()
    assert cfg.get("app.name") == "Test Incident Intelligence"


def test_config_resolve_default_dir_and_errors(temp_workspace: Path) -> None:
    """Verify default directory resolution when config_dir is None and error handling."""
    ConfigManager.reset()
    cfg = ConfigManager(config_dir=None)
    assert cfg is not None

    # Test load_yaml exceptions
    with pytest.raises(FileNotFoundError):
        cfg._load_yaml(Path(temp_workspace / "non_existent.yaml"))

    bad_yaml = temp_workspace / "config" / "bad.yaml"
    with open(bad_yaml, "w", encoding="utf-8") as f:
        f.write("key: [unclosed list\n  bad_syntax: :")

    with pytest.raises(ValueError):
        cfg._load_yaml(bad_yaml)
