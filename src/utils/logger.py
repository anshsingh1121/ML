"""
Logging Factory — Enterprise Structured Logging Setup.

Provides centralized logging configuration for the entire application.
Supports both human-readable and JSON-structured log formats.

Design Decisions:
    - JSON Formatter: Enterprise systems require machine-parseable logs for
      log aggregation (Splunk, ELK). JSON format enables this without agents.
    - Rotating File Handlers: Prevents disk exhaustion on long-running services.
      10MB per file, 5 backups = max 60MB disk usage per log category.
    - Separate Log Files: app.log, ml_training.log, api_access.log, error.log
      allow independent monitoring and retention policies.
    - Module-based Loggers: Each module gets its own logger via __name__,
      enabling granular log level control per component.

Usage:
    from src.utils.logger import LoggerFactory

    logger = LoggerFactory.get_logger(__name__)
    logger.info("Model training started", extra={"model": "rf_v1", "samples": 10000})
"""

import json
import logging
import logging.config
import logging.handlers
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from src.utils import robust_open

import yaml


class JsonFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.

    Produces one JSON object per log line, suitable for ingestion by
    enterprise log aggregation systems (Splunk, ELK, Datadog).
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            JSON-formatted log string.
        """
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # Include exception info if present
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Include extra fields (e.g., model_name, duration, incident_id)
        standard_attrs = {
            "name", "msg", "args", "created", "relativeCreated",
            "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "module", "filename", "levelno", "levelname", "pathname",
            "thread", "threadName", "process", "processName", "message",
            "msecs", "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                log_entry[key] = value

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class LoggerFactory:
    """
    Factory for creating and configuring application loggers.

    Loads logging configuration from YAML and ensures log directories
    exist before any handlers attempt to write.
    """

    _configured: bool = False

    @classmethod
    def configure(
        cls,
        config_path: Optional[str] = None,
        log_dir: Optional[str] = None,
    ) -> None:
        """
        Configure the logging system from a YAML configuration file.

        This should be called once at application startup. Subsequent calls
        are no-ops unless force=True.

        Args:
            config_path: Path to logging.yaml. Auto-detected if None.
            log_dir: Override log directory. Uses config value if None.
        """
        if cls._configured:
            return

        # Resolve config path
        config_file = cls._find_config(config_path)

        # Ensure log directories exist
        effective_log_dir = log_dir or "logs"
        Path(effective_log_dir).mkdir(parents=True, exist_ok=True)

        if config_file and config_file.exists():
            cls._configure_from_yaml(config_file, effective_log_dir)
        else:
            cls._configure_fallback(effective_log_dir)

        cls._configured = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a named logger, configuring the system if not yet done.

        Args:
            name: Logger name (typically __name__ of the calling module).

        Returns:
            Configured logging.Logger instance.

        Usage:
            logger = LoggerFactory.get_logger(__name__)
            logger.info("Processing started")
        """
        if not cls._configured:
            cls.configure()

        return logging.getLogger(name)

    @classmethod
    def _find_config(cls, config_path: Optional[str]) -> Optional[Path]:
        """Locate the logging configuration file."""
        if config_path:
            return Path(config_path)

        # Search from this file's location upward
        current = Path(__file__).resolve()
        for parent in current.parents:
            candidate = parent / "config" / "logging.yaml"
            if candidate.exists():
                return candidate

        # Fallback: CWD
        cwd_candidate = Path.cwd() / "config" / "logging.yaml"
        return cwd_candidate if cwd_candidate.exists() else None

    @classmethod
    def _configure_from_yaml(cls, config_path: Path, log_dir: str) -> None:
        """
        Load logging configuration from YAML file.

        Resolves relative log file paths against the project root.
        """
        try:
            with robust_open(config_path, "r") as f:
                log_config = yaml.safe_load(f)

            # Ensure log directories exist for all file handlers
            if "handlers" in log_config:
                for handler_name, handler_cfg in log_config["handlers"].items():
                    if "filename" in handler_cfg:
                        log_file = Path(handler_cfg["filename"])
                        log_file.parent.mkdir(parents=True, exist_ok=True)

            # Apply the configuration
            logging.config.dictConfig(log_config)

        except Exception as e:
            # If YAML config fails, fall back to basic config
            cls._configure_fallback(log_dir)
            logging.getLogger(__name__).warning(
                f"Failed to load logging config from {config_path}: {e}. "
                "Using fallback configuration."
            )

    @classmethod
    def _configure_fallback(cls, log_dir: str) -> None:
        """Configure basic logging as a fallback."""
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.handlers.RotatingFileHandler(
                    filename=os.path.join(log_dir, "app.log"),
                    maxBytes=10485760,
                    backupCount=5,
                    encoding="utf-8",
                ),
            ],
        )

    @classmethod
    def reset(cls) -> None:
        """
        Reset the logging configuration state.

        Used exclusively in testing to ensure clean state between tests.
        """
        root = logging.getLogger()
        for handler in list(root.handlers):
            try:
                handler.close()
                root.removeHandler(handler)
            except Exception:
                pass
        cls._configured = False


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function for getting a configured logger.

    This is the primary entry point for logging throughout the application.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured logging.Logger instance.

    Usage:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
    """
    return LoggerFactory.get_logger(name)
