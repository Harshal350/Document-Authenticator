"""Database engine, session factory and helper dependencies.

The application uses a SQLite database stored at ``<project_root>/data/
docuguard.db`` (configurable through ``database_url``).  This module exposes:

* ``Base``             - the declarative base every ORM model inherits from.
* ``engine``           - the SQLAlchemy engine (with ``PRAGMA foreign_keys``).
* ``SessionLocal``     - a thread-safe session factory.
* ``get_db()``         - FastAPI dependency that yields a scoped session.
* ``init_db()``        - creates tables that are not yet present.
"""

from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

_settings = get_settings()


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""

    def _asdict(self) -> dict:
        """Return a plain dictionary of the model's column values."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


def _sqlite_file_path(url: str) -> Optional[Path]:
    """Extract the filesystem path from a ``sqlite:///`` URL, if applicable."""
    if url.startswith("sqlite:///"):
        raw = url[len("sqlite:///"):]
        return Path(raw)
    return None


def _is_sqlite(url: Optional[str]) -> bool:
    return bool(url and url.startswith("sqlite"))


def _build_engine(url: str, echo: bool, pool_pre_ping: bool) -> Engine:
    """Create an engine, tuning SQLite-specific options."""
    kwargs: dict = {
        "echo": echo,
        "pool_pre_ping": pool_pre_ping,
    }

    if _is_sqlite(url):
        # SQLite does not support pooled connections across threads by default.
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_size"] = 5
        kwargs["max_overflow"] = 10

    engine = create_engine(url, **kwargs)

    if _is_sqlite(url):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, connection_record) -> None:  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
            finally:
                cursor.close()

    return engine


engine = _build_engine(_settings.database_url, echo=_settings.db_echo, pool_pre_ping=_settings.db_pool_pre_ping)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_engine() -> Engine:
    """Return the configured SQLAlchemy engine (useful after lazy re-config)."""
    return engine


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a database session.

    The session is always closed once the request is finished, even when an
    exception is raised mid-request.
    """
    db: Optional[Session] = None
    try:
        db = SessionLocal()
        yield db
    finally:
        if db is not None:
            db.close()


def init_db() -> None:
    """Create all missing tables and verify the schema is usable.

    The models are imported at call time so that every ``__tablename__`` is
    registered on ``Base.metadata`` before ``create_all`` runs.
    """
    import app.models.database_models  # noqa: F401  (registers models)

    sqlite_path = _sqlite_file_path(_settings.database_url)
    if sqlite_path is not None:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    Base.metadata.create_all(bind=engine)
    logger.info("Database initialised (engine=%s)", _settings.database_url)