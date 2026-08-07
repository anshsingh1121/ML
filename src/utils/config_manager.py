"""
Configuration Manager — Singleton YAML Configuration Loader.

Provides centralized, type-safe access to all application configuration.
Supports environment variable interpolation using ${VAR:default} syntax.

Design Decisions:
    - Singleton Pattern: One config instance shared across the entire application.
      Prevents redundant file I/O and ensures consistency.
    - Environment Variable Override: Secrets and environment-specific values
      (URLs, credentials) are injected via env vars, never stored in YAML.
    - Immutable After Load: Config is loaded once at startup. Explicit reload()
      is required to pick up changes (prevents mid-request config drift).
    - Dot-notation Access: config.get("ml.random_forest.n_estimators") for
      clean, readable access to nested values.

Usage:
    from src.utils.config_manager import ConfigManager

    config = ConfigManager()
    db_host = config.get("database.host", default="localhost")
    ml_config = config.get_section("ml")
"""

import os
import re
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional
from src.utils import robust_open

import yaml


class ConfigManager:
    """
    Thread-safe Singleton configuration manager.

    Loads YAML configuration files and provides dot-notation access
    to nested configuration values with environment variable interpolation.
    """

    _instance: Optional["ConfigManager"] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls, config_dir: Optional[str] = None) -> "ConfigManager":
        """Ensure only one instance exists (thread-safe Singleton)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls, config_dir: Optional[str] = None) -> "ConfigManager":
        """Return the Singleton instance of ConfigManager."""
        return cls(config_dir=config_dir)

    def __init__(self, config_dir: Optional[str] = None) -> None:
        """
        Initialize the configuration manager.

        Args:
            config_dir: Path to the configuration directory.
                        Defaults to 'config/' relative to project root.
        """
        if ConfigManager._initialized:
            return

        self._config: Dict[str, Any] = {}
        self._config_dir = self._resolve_config_dir(config_dir)

        self._load_all_configs()
        ConfigManager._initialized = True

    @staticmethod
    def _resolve_config_dir(config_dir: Optional[str]) -> Path:
        """Resolve the configuration directory path."""
        if config_dir:
            return Path(config_dir)

        # Walk up from this file to find the project root (where config/ lives)
        current = Path(__file__).resolve()
        for parent in current.parents:
            candidate = parent / "config"
            if candidate.is_dir():
                return candidate

        # Fallback: relative to CWD
        return Path.cwd() / "config"

    def _load_all_configs(self) -> None:
        """Load all YAML configuration files from the config directory."""
        config_files = [
            ("config.yaml", None),           # Merged into root
            ("model_config.yaml", "ml"),     # Merged under 'ml' key
            ("servicenow.yaml", "servicenow_api"),  # Merged under 'servicenow_api'
        ]

        for filename, namespace in config_files:
            filepath = self._config_dir / filename
            if filepath.exists():
                raw_config = self._load_yaml(filepath)
                if raw_config:
                    resolved = self._resolve_env_vars(raw_config)
                    if namespace:
                        self._config[namespace] = resolved
                    else:
                        self._config.update(resolved)

    def _load_yaml(self, filepath: Path) -> Dict[str, Any]:
        """
        Load and parse a YAML file.

        Args:
            filepath: Path to the YAML file.

        Returns:
            Parsed YAML content as a dictionary.

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the YAML is malformed.
        """
        try:
            with robust_open(filepath, "r") as f:
                content = yaml.safe_load(f)
                return content if isinstance(content, dict) else {}
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found: {filepath}"
            )
        except yaml.YAMLError as e:
            raise ValueError(
                f"Invalid YAML in configuration file {filepath}: {e}"
            )

    def _resolve_env_vars(self, config: Any) -> Any:
        """
        Recursively resolve environment variable placeholders.

        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.

        Args:
            config: Configuration value (dict, list, or scalar).

        Returns:
            Configuration with environment variables resolved.
        """
        if isinstance(config, dict):
            return {
                key: self._resolve_env_vars(value)
                for key, value in config.items()
            }
        elif isinstance(config, list):
            return [self._resolve_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Pattern: ${VAR_NAME} or ${VAR_NAME:default}
            pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"
            matches = re.findall(pattern, config)
            result = config
            for var_name, default_value in matches:
                env_value = os.environ.get(var_name, default_value or "")
                placeholder = (
                    f"${{{var_name}:{default_value}}}"
                    if default_value
                    else f"${{{var_name}}}"
                )
                result = result.replace(placeholder, env_value)
            return result
        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot-notation.

        Args:
            key: Dot-separated key path (e.g., "ml.random_forest.n_estimators").
            default: Default value if key is not found.

        Returns:
            The configuration value, or default if not found.

        Examples:
            >>> config = ConfigManager()
            >>> config.get("app.name")
            'Incident Intelligence Platform'
            >>> config.get("ml.random_forest.n_estimators", 100)
            200
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section as a dictionary.

        Args:
            section: Dot-separated section path.

        Returns:
            Configuration section as a dictionary, or empty dict.
        """
        result = self.get(section, {})
        return result if isinstance(result, dict) else {}

    def has(self, key: str) -> bool:
        """
        Check if a configuration key exists.

        Args:
            key: Dot-separated key path.

        Returns:
            True if the key exists, False otherwise.
        """
        return self.get(key) is not None

    @property
    def all(self) -> Dict[str, Any]:
        """Return the complete configuration dictionary (read-only copy)."""
        return dict(self._config)

    def reload(self) -> None:
        """
        Reload all configuration files from disk.

        Use sparingly — typically only for testing or admin operations.
        """
        self._config = {}
        self._load_all_configs()

    @classmethod
    def reset(cls) -> None:
        """
        Reset the Singleton instance.

        Used exclusively in testing to ensure clean state between tests.
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False

    def get_hybrid_config(self) -> Dict[str, Any]:
        """
        Get the hybrid recommendation engine configuration section (`ml.hybrid`).
        Returns dictionary containing weights, bonuses, thresholds, and MTTR fusion settings.
        """
        return self.get_section("ml.hybrid")

    def __repr__(self) -> str:
        return (
            f"ConfigManager("
            f"config_dir='{self._config_dir}', "
            f"keys={list(self._config.keys())})"
        )

