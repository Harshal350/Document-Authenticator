"""Upload endpoints for the DocuGuard API.

Single and batch document uploads are validated against the configured set of
allowed extensions and the maximum upload size before the file bytes are
persisted into the temporary upload directory and a ``documents`` row is
created. Content extraction is intentionally lazy: the raw file is only
processed (text / metadata / stats) once an analysis is triggered.
"""

from pathlib import Path
from typing import List, Tuple

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.document_processor import DocumentProcessor
from app.database.database import get_db
from app.models.database_models import Document
from app.models.schemas import BatchUploadResponse, UploadResponse
from app.utils.file_validation import (
    generate_temp_path,
    get_extension,
    sanitize_filename,
    validate_file_size,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter()

_document_processor = DocumentProcessor()


async def _persist_upload(file: UploadFile) -> Tuple[Path, int, str, str]:
    """Validate an uploaded file and write it to the temporary upload dir.

    Returns:
        A tuple ``(stored_path, size_bytes, extension, sanitized_name)``.

    Raises:
        HTTPException: when the file fails any validation rule or cannot be
            written to disk.
    """
    original_name = file.filename or "document"
    try:
        ext = get_extension(original_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if ext not in settings.allowed_extensions_set:
        allowed = ", ".join(sorted(settings.allowed_extensions_set))
        raise HTTPException(
            status_code=400,
            detail=(
                f"File extension '.{ext}' is not allowed. "
                f"Permitted extensions: {allowed}"
            ),
        )

    content = await file.read()
    try:
        validate_file_size(len(content), settings.max_upload_size_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    stored_path = generate_temp_path(original_name)
    try:
        stored_path.write_bytes(content)
    except OSError as exc:
        logger.error("Could not store upload %s: %s", original_name, exc)
        raise HTTPException(
            status_code=500,
            detail="The uploaded file could not be stored on the server",
        )

    logger.info(
        "Stored upload '%s' (%.2f KB) -> %s",
        original_name,
        len(content) / 1024.0,
        stored_path,
    )
    return stored_path, len(content), ext, sanitize_filename(original_name)


def _register_document(
    stored_path: Path,
    file_size: int,
    ext: str,
    safe_name: str,
    db: Session,
) -> Document:
    """Process the stored file and create the ``documents`` row.

    If processing fails, the temporary file is removed before the HTTP error
    is raised so that orphaned uploads are not left behind.
    """
    result = _document_processor.process_document(str(stored_path), ext)
    if not result.get("success"):
        stored_path.unlink(missing_ok=True)
        logger.error(
            "Rejected upload '%s': %s",
            safe_name,
            result.get("error") or "processing failed",
        )
        raise HTTPException(
            status_code=400,
            detail=result.get("error") or "Failed to process the uploaded document",
        )

    stats = result.get("stats") or {}
    document = Document(
        filename=stored_path.name,
        original_filename=safe_name,
        file_type=ext,
        file_size=file_size,
        pages=int(stats.get("pages", 0)) if stats.get("pages") is not None else 0,
        words=int(stats.get("words", 0)) if stats.get("words") is not None else 0,
        characters=(
            int(stats.get("characters", 0))
            if stats.get("characters") is not None
            else 0
        ),
        tables_detected=(
            int(stats.get("tables_detected", 0))
            if stats.get("tables_detected") is not None
            else 0
        ),
        ocr_used=bool(stats.get("ocr_used", False)),
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    logger.info("Registered document #%s (%s)", document.id, safe_name)
    return document


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> UploadResponse:
    """Upload a single document for later forensic analysis.

    The file is validated (extension + size), stored under a unique temporary
    name and registered in the database. Processing and risk analysis are
    performed separately via :ref:`POST /analyze/{document_id}`.
    """
    stored_path, file_size, ext, safe_name = await _persist_upload(file)
    document = _register_document(stored_path, file_size, ext, safe_name, db)

    return UploadResponse(
        message="File uploaded successfully",
        document_id=document.id,
        filename=safe_name,
    )


@router.post("/upload/batch", response_model=BatchUploadResponse)
async def upload_batch(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> BatchUploadResponse:
    """Upload multiple documents in a single request.

    Each file is validated and registered independently so that a single bad
    file does not discard the rest of the batch. Failures are reported in
    ``failed_files`` with the per-file reason.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided")

    uploaded_count = 0
    failed_files: List[dict] = []

    for file in files:
        original_name = file.filename or "document"
        try:
            stored_path, file_size, ext, safe_name = await _persist_upload(file)
            _register_document(stored_path, file_size, ext, safe_name, db)
            uploaded_count += 1
        except HTTPException as exc:
            detail = exc.detail
            failed_files.append(
                {
                    "filename": original_name,
                    "error": detail if isinstance(detail, str) else str(detail),
                }
            )
            logger.warning("Batch upload failed for '%s': %s", original_name, detail)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unexpected batch upload failure for '%s'", original_name)
            failed_files.append({"filename": original_name, "error": str(exc)})

    return BatchUploadResponse(
        message=(
            f"Uploaded {uploaded_count} of {len(files)} files successfully"
            if failed_files
            else f"All {uploaded_count} files uploaded successfully"
        ),
        uploaded_count=uploaded_count,
        failed_count=len(failed_files),
        failed_files=failed_files,
    )