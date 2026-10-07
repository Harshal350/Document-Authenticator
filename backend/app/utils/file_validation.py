"""File validation and sanitisation utilities for uploaded documents.

Functions in this module perform extension checks, MIME sniffing (with a
fallback to python-magic when available), size enforcement, filename
sanitisation and unique temporary-path generation.
"""

import mimetypes
import os
import re
import uuid
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, Union

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Constant: allowed extensions
# --------------------------------------------------------------------------- #
ALLOWED_EXTENSIONS: Set[str] = {
    "pdf",
    "doc",
    "docx",
    "png",
    "jpg",
    "jpeg",
    "tif",
    "tiff",
    "bmp",
}

# --------------------------------------------------------------------------- #
# MIME-type mapping (extension -> acceptable MIME prefixes / exact types)
# --------------------------------------------------------------------------- #
_MIME_MAP: Dict[str, Tuple[str, ...]] = {
    "pdf": ("application/pdf",),
    "doc": ("application/msword",),
    "docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",  # DOCX is a ZIP archive; python-magic sometimes reports application/zip
    ),
    "png": ("image/png",),
    "jpg": ("image/jpeg",),
    "jpeg": ("image/jpeg",),
    "tif": ("image/tiff",),
    "tiff": ("image/tiff",),
    "bmp": ("image/bmp",),
}

# --------------------------------------------------------------------------- #
# Magic-byte signatures (used when python-magic is unavailable)
# --------------------------------------------------------------------------- #
_SIGNATURES: Tuple[bytes, Tuple[str, ...]] = (
    (b"%PDF", ("application/pdf",)),
    (b"\x89PNG\r\n\x1a\n", ("image/png",)),
    (b"\xff\xd8\xff", ("image/jpeg",)),
    (b"II\x2a\x00", ("image/tiff",)),   # little-endian TIFF
    (b"MM\x00\x2a", ("image/tiff",)),   # big-endian TIFF
    (b"BM", ("image/bmp",)),
    (b"PK\x03\x04", ("application/zip", "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
)

# Lazy import of python-magic (optional dependency)
_magic_module = None
_magic_import_attempted = False


def _get_magic_module():
    """Return the python-magic module, or ``None`` if unavailable."""
    global _magic_module, _magic_import_attempted
    if not _magic_import_attempted:
        _magic_import_attempted = True
        try:
            import magic as _mod

            _magic_module = _mod
            logger.debug("python-magic available for MIME detection")
        except ImportError:
            logger.debug("python-magic not installed; using signature-based MIME fallback")
    return _magic_module


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #
def get_extension(filename: str) -> str:
    """Return the normalised (lowercase, dot-stripped) extension from ``filename``.

    Raises ``ValueError`` when the filename has no extension.
    """
    if "." not in filename:
        raise ValueError(f"File '{filename}' has no extension")
    ext = filename.rsplit(".", 1)[-1].strip().lower()
    if not ext:
        raise ValueError(f"File '{filename}' has an empty extension")
    return ext


def validate_file_extension(filename: str) -> str:
    """Validate and return the normalised extension.

    :raises ValueError: if the extension is not in :data:`ALLOWED_EXTENSIONS`.
    """
    ext = get_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"File extension '.{ext}' is not allowed. "
            f"Permitted extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return ext


def validate_file_size(file_size: int, max_size: Optional[int] = None) -> int:
    """Validate *file_size* against *max_size* (defaults to the configured upload limit).

    :raises ValueError: when *file_size* is outside the allowed range.
    :returns: the validated *file_size*.
    """
    if file_size < 0:
        raise ValueError("File size must not be negative")
    if file_size == 0:
        raise ValueError("File is empty")
    if max_size is None:
        max_size = get_settings().max_upload_size_bytes
    if file_size > max_size:
        mb = round(file_size / (1024 * 1024), 2)
        max_mb = round(max_size / (1024 * 1024), 2)
        raise ValueError(f"File size {mb} MB exceeds the maximum allowed size of {max_mb} MB")
    return file_size


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """Produce a filesystem-safe version of *filename*.

    * Removes path separators and ``..`` traversals.
    * Replaces anything that is not alphanumeric, dash, underscore or dot.
    * Falls back to ``document_<uuid>`` when the stem becomes empty.
    * Truncates the result to *max_length* characters.
    """
    if not filename or not filename.strip():
        return f"document_{uuid.uuid4().hex[:12]}"

    filename = filename.replace("\\", "/").split("/")[-1]

    name_part, *ext_parts = filename.rsplit(".", 1) if "." in filename else (filename,)
    ext = f".{ext_parts[0].lower()}" if ext_parts else ""

    name_part = re.sub(r"[^\w\-.]", "_", name_part)
    name_part = re.sub(r"_{2,}", "_", name_part).strip("_-")
    if not name_part:
        name_part = f"document_{uuid.uuid4().hex[:12]}"

    if len(name_part) + len(ext) > max_length:
        max_name = max_length - len(ext)
        name_part = name_part[:max_name]

    return f"{name_part}{ext}"


def generate_temp_path(
    original_filename: str,
    upload_dir: Optional[str] = None,
) -> Path:
    """Return a unique temporary path inside *upload_dir*.

    The original extension is preserved while the name is replaced by a UUID4
    hex string so that collisions are impossible.  The parent directory is
    created automatically when it does not yet exist.
    """
    settings = get_settings()
    directory = Path(upload_dir) if upload_dir else settings.temp_dir_path

    ext = ""
    if "." in original_filename:
        ext = f".{original_filename.rsplit('.', 1)[-1].strip().lower()}"

    unique_name = f"{uuid.uuid4().hex}{ext}"
    temp_path = directory / unique_name
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    return temp_path


def detect_mime_type(file_path: Union[str, Path]) -> str:
    """Detect the MIME type of a file.

    Tries ``python-magic`` first; falls back to signature sniffing; finally
    falls back to :func:`mimetypes.guess_type`.
    """
    path = Path(file_path)
    if not path.is_file():
        return "application/octet-stream"

    # 1. python-magic
    magic_mod = _get_magic_module()
    if magic_mod is not None:
        try:
            mime = magic_mod.from_file(str(path), mime_type=True)
            if mime:
                return mime
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("python-magic detection failed: %s", exc)

    # 2. signature sniffing
    try:
        with open(path, "rb") as fh:
            header = fh.read(32)
        for sig, mime_candidates in _SIGNATURES:
            if header.startswith(sig):
                return mime_candidates[0]
    except OSError:
        pass

    # 3. stdlib mimetypes
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def validate_mime_type(
    filename: str,
    detected_mime: Optional[str] = None,
    file_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Return ``True`` when *detected_mime* is acceptable for the given extension.

    If *detected_mime* is not provided the function will attempt to detect it
    from *file_path* or from the filename's extension mapping.
    """
    ext = get_extension(filename)
    expected_mimes = _MIME_MAP.get(ext, ())

    if detected_mime is None and file_path is not None:
        detected_mime = detect_mime_type(file_path)

    if detected_mime is None:
        guessed, _ = mimetypes.guess_type(filename)
        detected_mime = guessed or ""

    detected_lower = detected_mime.lower()
    for candidate in expected_mimes:
        if detected_lower.startswith(candidate.lower()):
            return True

    if not expected_mimes:
        return True

    logger.warning(
        "MIME mismatch for '%s': expected one of %s, detected %s",
        filename,
        expected_mimes,
        detected_mime,
    )
    return False


def validate_upload(
    original_filename: str,
    file_size: int,
    file_path: Optional[Union[str, Path]] = None,
    upload_dir: Optional[str] = None,
) -> Path:
    """Run the full validation pipeline and return the safe temp path.

    This combines extension validation, size validation, filename sanitisation,
    MIME detection, and unique-path generation in a single call.

    :param original_filename: the filename sent by the client.
    :param file_size: the reported file size in bytes.
    :param file_path: optional path to the file on disk for MIME sniffing.
    :param upload_dir: optional override for the upload directory.
    :raises ValueError: on any validation failure.
    :returns: the unique ``Path`` where the file should be stored.
    """
    ext = validate_file_extension(original_filename)
    validate_file_size(file_size)

    safe_name = sanitize_filename(original_filename)

    if file_path is not None:
        mime = detect_mime_type(Path(file_path) if isinstance(file_path, str) else file_path)
        if not validate_mime_type(original_filename, detected_mime=mime):
            raise ValueError(
                f"MIME type '{mime}' is not permitted for a '.{ext}' file"
            )

    return generate_temp_path(safe_name, upload_dir=upload_dir)