"""Application configuration for the DocuGuard backend.

All settings are defined as a ``pydantic-settings`` ``BaseSettings`` subclass so
they can be sourced from environment variables or from a ``.env`` file placed at
the project root.  The ``Settings`` instance is created once and cached via
``functools.lru_cache`` so every module in the application shares the same
configuration.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Set

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.utils.logging import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Filesystem layout helpers.
#
# config.py lives at: backend/app/config.py
#   APP_DIR       -> backend/app
#   BACKEND_DIR   -> backend
#   PROJECT_ROOT  -> DocGuard (repository root)
# --------------------------------------------------------------------------- #
APP_DIR: Path = Path(__file__).resolve().parent
BACKEND_DIR: Path = APP_DIR.parent
PROJECT_ROOT: Path = BACKEND_DIR.parent

_DATA_DIR: Path = PROJECT_ROOT / "data"
_RAW_DIR: Path = _DATA_DIR / "raw"
_PROCESSED_DIR: Path = _DATA_DIR / "processed"
_MODEL_DIR: Path = PROJECT_ROOT / "ml_models"
_REPORT_DIR: Path = PROJECT_ROOT / "reports"

_DEFAULT_ALLOWED_EXTENSIONS: List[str] = [
    "pdf",
    "doc",
    "docx",
    "png",
    "jpg",
    "jpeg",
    "tif",
    "tiff",
    "bmp",
]


class Settings(BaseSettings):
    """Runtime configuration for DocuGuard.

    Values are read from environment variables with a ``.env`` file at the
    project root acting as the default source.  Environment variables take
    precedence over the ``.env`` file which in turn overrides the defaults
    declared here.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- Application -------------------------------------------------------- #
    app_name: str = "DocuGuard"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    api_prefix: str = "/api/v1"
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
            "http://localhost:5175",
            "http://127.0.0.1:5175",
        ]
    )

    # -- Database ------------------------------------------------------------- #
    database_url: str = f"sqlite:///{_DATA_DIR / 'docuguard.db'}"
    db_echo: bool = False
    db_pool_pre_ping: bool = True

    # -- Uploads -------------------------------------------------------------- #
    upload_max_size_mb: int = 25
    allowed_extensions: List[str] = Field(default_factory=lambda: list(_DEFAULT_ALLOWED_EXTENSIONS))
    temp_dir: str = str(_RAW_DIR)
    processed_dir: str = str(_PROCESSED_DIR)

    # -- Machine learning models ------------------------------------------------- #
    model_dir: str = str(_MODEL_DIR)
    lr_model_path: str = str(_MODEL_DIR / "logistic_model.joblib")
    rf_model_path: str = str(_MODEL_DIR / "rf_model.joblib")
    xgb_model_path: str = str(_MODEL_DIR / "xgb_model.joblib")
    feature_columns_path: str = str(_MODEL_DIR / "feature_columns.joblib")
    scaler_path: str = str(_MODEL_DIR / "scaler.joblib")
    enable_ml: bool = True

    # -- Risk thresholds (risk scores are normalised to 0-100) ------------------- #
    # Decision bands (mirrors the product spec and the RiskEngine):
    #   0-29 -> Likely Genuine | 30-59 -> Suspicious | 60-100 -> Likely Fake
    high_risk_threshold: float = 60.0
    medium_risk_threshold: float = 30.0

    # -- OCR --------------------------------------------------------------------- #
    enable_ocr: bool = False
    ocr_engine: str = "tesseract"
    tesseract_path: Optional[str] = None
    ocr_languages: str = "eng"

    # -- Feature extraction ----------------------------------------------------- #
    max_pages_to_process: int = 500
    enable_image_forensics: bool = True
    enable_layout_analysis: bool = True

    # -- Reports ----------------------------------------------------------------- #
    report_dir: str = str(_REPORT_DIR)
    report_max_age_days: int = 30

    # -- Logging & security ------------------------------------------------------- #
    log_level: str = "INFO"
    log_file: Optional[str] = str(_DATA_DIR / "docuguard.log")
    secret_key: str = "change-me-in-production"
    token_expiry_minutes: int = 60

    # --------------------------------------------------------------------------- #
    # Validators
    # --------------------------------------------------------------------------- #
    @field_validator("upload_max_size_mb")
    @classmethod
    def _validate_upload_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("upload_max_size_mb must be a positive integer")
        return value

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def _parse_allowed_extensions(cls, value) -> List[str]:
        if isinstance(value, str):
            parts = [p.strip().lower().lstrip(".") for p in value.split(",") if p.strip()]
            if not parts:
                raise ValueError("allowed_extensions must not be empty")
            return parts
        if isinstance(value, (list, tuple, set)):
            parts = [p.strip().lower().lstrip(".") for p in value if p.strip()]
            if not parts:
                raise ValueError("allowed_extensions must not be empty")
            return parts
        raise ValueError("allowed_extensions must be a string or a list of strings")

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        normalized = value.upper()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in valid:
            raise ValueError(f"log_level must be one of {sorted(valid)}")
        return normalized

    @field_validator("high_risk_threshold", "medium_risk_threshold")
    @classmethod
    def _validate_thresholds(cls, value: float) -> float:
        if not 0.0 <= value <= 100.0:
            raise ValueError("risk thresholds must be between 0 and 100")
        return value

    # --------------------------------------------------------------------------- #
    # Derived helpers
    # --------------------------------------------------------------------------- #
    @property
    def max_upload_size_bytes(self) -> int:
        """Maximum allowed upload size expressed in bytes (default 25 MB)."""
        return self.upload_max_size_mb * 1024 * 1024

    @property
    def allowed_extensions_set(self) -> Set[str]:
        """The allowed extensions as a set for fast membership checks."""
        return set(self.allowed_extensions)

    @property
    def temp_dir_path(self) -> Path:
        return Path(self.temp_dir)

    @property
    def processed_dir_path(self) -> Path:
        return Path(self.processed_dir)

    @property
    def model_dir_path(self) -> Path:
        return Path(self.model_dir)

    @property
    def report_dir_path(self) -> Path:
        return Path(self.report_dir)

    def ensure_directories(self) -> None:
        """Create every directory the application depends on, if missing."""
        for directory in (
            self.temp_dir_path,
            self.processed_dir_path,
            self.model_dir_path,
            self.report_dir_path,
            Path(self.database_url.replace("sqlite:///", "")).parent
            if self.database_url.startswith("sqlite")
            else PROJECT_ROOT,
        ):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:  # pragma: no cover - defensive
                logger.warning("Could not create directory %s: %s", directory, exc)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application ``Settings`` singleton."""
    settings = Settings()
    settings.ensure_directories()
    logger.debug(
        "Loaded settings for %s v%s (environment=%s)",
        settings.app_name,
        settings.app_version,
        settings.environment,
    )
    return settings