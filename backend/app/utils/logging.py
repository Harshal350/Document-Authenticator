"""Centralised logging configuration for the DocuGuard backend.

This module configures the standard library ``logging`` package once per
process: logs are emitted both to ``stdout`` (for development / containers) and
to a rotating file ``<data>/docuguard.log`` (when a log file path is
configured).  Every module in the application obtains a named logger through
:func:`get_logger`, which guarantees the configuration has been applied.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock

# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Loggers that are extremely chatty at DEBUG/INFO and are filtered out during
# normal operation to keep the log useful.
_NOISY_LOGGERS: tuple = (
    "sqlalchemy.engine",
    "urllib3",
    "watchfiles",
)

# -- defaults pulled from the environment without importing app.config (this
# -- keeps the logging module dependency-free and avoids circular imports).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
_DEFAULT_LOG_DIR = os.path.join(_PROJECT_ROOT, "data")

_env_level = os.getenv("DOCUGUARD_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO")).upper()
_LOG_LEVEL: str = _env_level if _env_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL") else "INFO"
_LOG_FILE: str = os.getenv("DOCUGUARD_LOG_FILE", os.getenv("LOG_FILE", os.path.join(_DEFAULT_LOG_DIR, "docuguard.log")))
_MAX_BYTES: int = 5 * 1024 * 1024  # 5 MB per file
_BACKUP_COUNT: int = 5
_KEEP_NOISY_LOGGERS: bool = os.getenv("DOCUGUARD_DEBUG_LOGGING", "").lower() in ("1", "true", "yes")

_configured: bool = False
_lock: Lock = Lock()


class _TruncatingFilter(logging.Filter):
    """Caps the length of an individual log record's message."""

    def __init__(self, max_length: int = 20000) -> None:
        super().__init__()
        self.max_length = max_length

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
            if len(message) > self.max_length:
                record.msg = message[: self.max_length] + " ... [truncated]"
        except Exception:  # pragma: no cover - defensive
            pass
        return True


def _interpolate(value: str) -> str:
    """Expand ``~`` and environment variables embedded in a path string."""
    return os.path.expandvars(os.path.expanduser(value))


def configure_logging(
    log_level: str = _LOG_LEVEL,
    log_file: str = _LOG_FILE,
    max_bytes: int = _MAX_BYTES,
    backup_count: int = _BACKUP_COUNT,
    force: bool = False,
) -> logging.Logger:
    """Configure the root logger for the application.

    Safe to call multiple times; unless ``force`` is True subsequent calls are
    no-ops because the configuration is performed once per process.

    :param log_level: minimum level to record (DEBUG/INFO/WARNING/ERROR/CRITICAL).
    :param log_file: path of the rotating log file, or an empty string/None to
        disable file logging.
    :param max_bytes: maximum size of a single log file before rotation.
    :param backup_count: number of rotated backups to retain.
    :param force: re-apply the configuration even if already configured.
    """
    global _configured

    with _lock:
        if _configured and not force:
            return logging.getLogger()

        level: int = getattr(logging, str(log_level).upper(), logging.INFO)
        root = logging.getLogger()
        root.setLevel(level)

        # Remove any handlers installed by a previous configuration to avoid
        # duplicate lines when force=True.
        for handler in list(root.handlers):
            root.removeHandler(handler)

        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

        file_path = _interpolate(log_file) if log_file else None
        if file_path:
            try:
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                file_handler = RotatingFileHandler(
                    file_path,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                root.addHandler(file_handler)
            except OSError as exc:
                root.warning("File logging disabled, could not create %s: %s", file_path, exc)

        truncating_filter = _TruncatingFilter()
        for handler in root.handlers:
            handler.addFilter(truncating_filter)

        if not _KEEP_NOISY_LOGGERS:
            for logger_name in _NOISY_LOGGERS:
                logging.getLogger(logger_name).setLevel(logging.WARNING)

        _configured = True
        root.log(level, "Logging configured (level=%s, file=%s)", log_level.upper(), file_path or "console only")
        return root


def get_logger(name: str = __name__) -> logging.Logger:
    """Return a configured logger for the module identified by ``name``."""
    if not _configured:
        configure_logging()
    return logging.getLogger(name)