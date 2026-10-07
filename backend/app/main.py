"""DocuGuard FastAPI application entry point.

Wires up the five API routers under the ``/api/v1`` prefix, configures CORS
for the local frontend, initialises the database during startup, mounts the
generated reports as static files and exposes a health-check endpoint.
"""

import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app.api.analysis import router as analysis_router
from app.api.history import router as history_router
from app.api.models import router as models_router
from app.api.reports import router as reports_router
from app.api.upload import router as upload_router
from app.config import get_settings
from app.database.database import engine, init_db
from app.models.schemas import HealthResponse
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle: initialise the database on boot."""
    logger.info(
        "Starting %s v%s (environment=%s)",
        settings.app_name,
        settings.app_version,
        settings.environment,
    )
    init_db()
    logger.info("Database ready at %s", settings.database_url)
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    description=(
        "AI-assisted forensic document analysis API. Upload documents, run "
        "fraud-risk analyses and generate PDF reports."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url=f"{settings.api_prefix}/docs",
    openapi_url=f"{settings.api_prefix}/openapi.json",
    redoc_url=f"{settings.api_prefix}/redoc",
)

# --------------------------------------------------------------------------- #
# CORS - allow the local frontend to call the API.
# --------------------------------------------------------------------------- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = settings.api_prefix

# --------------------------------------------------------------------------- #
# Routers
# --------------------------------------------------------------------------- #
app.include_router(upload_router, prefix=API_PREFIX, tags=["upload"])
app.include_router(analysis_router, prefix=API_PREFIX, tags=["analysis"])
app.include_router(history_router, prefix=API_PREFIX, tags=["history"])
app.include_router(reports_router, prefix=API_PREFIX, tags=["reports"])
app.include_router(models_router, prefix=API_PREFIX, tags=["models"])

# --------------------------------------------------------------------------- #
# Reports served as static files.
# --------------------------------------------------------------------------- #
settings.report_dir_path.mkdir(parents=True, exist_ok=True)
app.mount(
    f"{API_PREFIX}/reports",
    StaticFiles(directory=str(settings.report_dir_path)),
    name="reports",
)


@app.get(f"{API_PREFIX}/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return basic service health, including a live database check."""
    database_status = "connected"
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:
        logger.error("Health check: database unreachable: %s", exc)
        database_status = "disconnected"

    return HealthResponse(
        status="healthy" if database_status == "connected" else "degraded",
        version=settings.app_version,
        database=database_status,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.get("/")
async def root() -> Dict[str, Any]:
    """Return basic service metadata for the API root."""
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "docs": f"{API_PREFIX}/docs",
        "health": f"{API_PREFIX}/health",
    }