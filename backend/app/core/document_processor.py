"""Document processing pipeline for DocuGuard.

The :class:`DocumentProcessor` class ingests PDF, DOCX, TXT and image files,
extracts text and metadata, and reports structured statistics that downstream
fraud-detection modules consume. Every file format is handled defensively:
corrupted files, password-protected PDFs and missing OCR binaries produce
structured error results rather than raising exceptions.
"""

import logging
import re
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:  # pragma: no cover - environment specific
    import fitz
except Exception:
    fitz = None
    logger.debug("PyMuPDF (fitz) is not available; PDF processing disabled.")

try:  # pragma: no cover - environment specific
    import docx
except Exception:
    docx = None
    logger.debug("python-docx is not available; DOCX processing disabled.")

try:  # pragma: no cover - environment specific
    import cv2
except Exception:
    cv2 = None
    logger.debug("OpenCV (cv2) is not available; image preprocessing disabled.")

try:  # pragma: no cover - environment specific
    from PIL import Image as PILImage
    import numpy as np
except Exception:
    PILImage = None
    np = None

try:  # pragma: no cover - relative import safety
    from .ocr_engine import OCREngine
except ImportError:  # pragma: no cover - allows running as a standalone module
    from ocr_engine import OCREngine

try:  # pragma: no cover - relative import safety
    from app.config import get_settings
except Exception:  # pragma: no cover - standalone fallback
    get_settings = None

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
}
_MIN_TEXT_WORDS = 3
_PDF_DATE_PATTERN = re.compile(
    r"(?P<year>\d{4})(?P<month>\d{2})?(?P<day>\d{2})?"
    r"(?P<hour>\d{2})?(?P<minute>\d{2})?(?P<second>\d{2})?"
)
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _variance(values: list) -> float:
    """Population variance of a list of numbers (``0.0`` when degenerate)."""
    if not values or len(values) < 2:
        return 0.0
    try:
        return round(statistics.pvariance(values), 4)
    except (statistics.StatisticsError, TypeError, ValueError):
        return 0.0


class DocumentProcessor:
    """Coordinates extraction, OCR and statistics for supported documents.

    The processor delegates to dedicated private handlers for each file type
    and normalizes their output into a single result schema.
    """

    def __init__(self, ocr_engine: Optional[OCREngine] = None) -> None:
        """Initialize the processor.

        Args:
            ocr_engine: An :class:`OCREngine` instance; created lazily when
                omitted.
        """
        self.ocr_engine = ocr_engine or OCREngine()
        self.ocr_enabled: bool = self._resolve_ocr_flag()

    @staticmethod
    def _resolve_ocr_flag() -> bool:
        """Read the ``enable_ocr`` configuration flag (defaults to ``True``)."""
        if get_settings is None:  # pragma: no cover - standalone fallback
            return True
        try:
            return bool(get_settings().enable_ocr)
        except Exception:  # pragma: no cover - defensive
            return True

    def _ocr_page(self, page: Any) -> Tuple[str, float]:
        """OCR a single PDF page when OCR is enabled; otherwise no-op."""
        if not self.ocr_enabled:
            return "", 0.0
        return self.ocr_engine.handle_image_page(page)

    def _ocr_frame(self, frame: Any) -> Tuple[str, float]:
        """OCR a single image frame when OCR is enabled; otherwise no-op."""
        if not self.ocr_enabled:
            return "", 0.0
        return self.ocr_engine.perform_ocr(frame)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process_document(
        self, file_path: str, file_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a document and return extracted content and statistics.

        Args:
            file_path: Absolute or relative path to the file.
            file_type: Optional type hint (``pdf``, ``docx``, ``txt``, or an
                image extension). Inferred from the file suffix when omitted.

        Returns:
            A dictionary with ``success``, ``error``, ``text``, ``pages``,
            ``metadata``, ``stats`` and ``ocr_used`` keys. ``success`` is
            ``False`` when the file could not be processed.
        """
        start = time.time()
        path = Path(file_path)
        file_type = self._normalize_type(file_type, path)

        if not path.is_file():
            return self._error_result(
                path, file_type, "File not found: '{}'".format(str(path)), start
            )

        handler = self._get_handler(file_type)
        try:
            result = handler(path, file_type, start)
        except Exception:  # pragma: no cover - last-resort guard
            logger.exception(
                "Unhandled error while processing %s (%s)",
                path,
                file_type,
            )
            result = self._error_result(
                path, file_type, "Unexpected processing error", start
            )
        return result

    # ------------------------------------------------------------------ #
    # Dispatch helpers
    # ------------------------------------------------------------------ #
    def _get_handler(self, file_type: str) -> Any:
        """Resolve the private handler method for a normalized type."""
        if file_type in (PDF_TYPE, DOCX_TYPE, TEXT_TYPE, IMAGE_TYPE):
            return getattr(self, "_process_{}".format(file_type))
        return self._process_unknown

    def _normalize_type(
        self, file_type: Optional[str], path: Path
    ) -> str:
        """Normalize a caller-supplied type or infer one from the suffix."""
        if not file_type:
            return self._infer_type(path)
        normalized = file_type.strip().lstrip(".").lower()
        if normalized in {"pdf"}:
            return PDF_TYPE
        if normalized in {"docx"}:
            return DOCX_TYPE
        if normalized in {"txt", "text"}:
            return TEXT_TYPE
        if normalized.lstrip(".") in {
            extension.lstrip(".") for extension in IMAGE_EXTENSIONS
        }:
            return IMAGE_TYPE
        if normalized in {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"}:
            return IMAGE_TYPE
        return self._infer_type(path)

    def _infer_type(self, path: Path) -> str:
        """Infer the normalized type from a file's suffix."""
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return PDF_TYPE
        if suffix == ".docx":
            return DOCX_TYPE
        if suffix in {".txt", ".text", ".log", ".md"}:
            return TEXT_TYPE
        if suffix in IMAGE_EXTENSIONS:
            return IMAGE_TYPE
        return "unknown"

    # ------------------------------------------------------------------ #
    # PDF
    # ------------------------------------------------------------------ #
    def _process_pdf(
        self, file_path: Path, file_type: str, start: float
    ) -> Dict[str, Any]:
        """Extract text and metadata from a PDF.

        Pages without meaningful text are rendered and OCR'd when the page
        contains embedded images (scanned-document detection).
        """
        if fitz is None:
            return self._error_result(
                file_path, file_type, "PyMuPDF is not installed", start
            )

        document = None
        try:
            document = fitz.open(str(file_path))
        except Exception as exc:
            logger.warning("Could not open PDF %s: %s", file_path, exc)
            return self._error_result(
                file_path,
                file_type,
                "Corrupted or unreadable PDF: {}".format(exc),
                start,
            )

        try:
            if document.needs_pass:
                authenticated = False
                try:
                    authenticated = bool(document.authenticate(""))
                except Exception:
                    authenticated = False
                if not authenticated:
                    return self._error_result(
                        file_path,
                        file_type,
                        "PDF is password protected",
                        start,
                    )

            pages_url = []
            ocr_pages = 0
            image_only_pages = 0
            blank_pages = 0
            image_count = 0
            ocr_confidence_sum = 0.0
            ocr_confidence_count = 0
            font_sizes: list = []
            line_offsets: list = []

            for page in document:
                text = (page.get_text("text") or "").strip()
                try:
                    image_count += len(page.get_images(full=True))
                except Exception:
                    pass
                self._collect_layout(page, font_sizes, line_offsets)

                if len(text.split()) >= _MIN_TEXT_WORDS:
                    pages_url.append(text)
                    continue

                has_images = bool(page.get_images(full=True))
                ocr_text, confidence = self._ocr_page(page)
                if has_images:
                    image_only_pages += 1
                if ocr_text.strip():
                    ocr_pages += 1
                    pages_url.append(ocr_text)
                else:
                    pages_url.append(text)
                    if not text:
                        blank_pages += 1

                if confidence > 0.0:
                    ocr_confidence_sum += confidence
                    ocr_confidence_count += 1

            text = "\n\n".join(pages_url)
            metadata = self._extract_pdf_metadata(document)
            metadata["page_count"] = document.page_count
            metadata["image_only_pages"] = image_only_pages
            metadata["creation_date"] = metadata.get("creationDate")
            metadata["modification_date"] = metadata.get("modDate")

            ocr_confidence = (
                round(ocr_confidence_sum / ocr_confidence_count, 2)
                if ocr_confidence_count
                else 0.0
            )
            return self._finalize(
                file_path=file_path,
                file_type=file_type,
                text=text,
                pages=document.page_count,
                metadata=metadata,
                tables_detected=0,
                ocr_used=ocr_pages > 0,
                ocr_confidence=ocr_confidence,
                ocr_pages=ocr_pages,
                image_only_pages=image_only_pages,
                start=start,
                blank_page_count=blank_pages,
                image_count=image_count,
                font_size_variance=_variance(font_sizes),
                alignment_variance=_variance(line_offsets),
            )
        except Exception as exc:
            logger.exception("Failed to extract text from PDF %s", file_path)
            return self._error_result(
                file_path,
                file_type,
                "Failed to process PDF: {}".format(exc),
                start,
            )
        finally:
            if document is not None:
                try:
                    document.close()
                except Exception:
                    logger.debug("Failed to close PDF %s", file_path, exc_info=True)

    @staticmethod
    def _collect_layout(page: Any, font_sizes: list, line_offsets: list) -> None:
        """Collect font sizes and line start offsets from a PDF page's layout."""
        try:
            raw = page.get_text("dict")
        except Exception:
            return
        for block in raw.get("blocks", []) or []:
            for line in block.get("lines", []) or []:
                spans = line.get("spans", []) or []
                for span in spans:
                    size = span.get("size")
                    if size is not None:
                        font_sizes.append(float(size))
                x0 = line.get("bbox", [None])[0]
                if x0 is not None:
                    line_offsets.append(float(x0))

    def _extract_pdf_metadata(self, document: Any) -> Dict[str, Any]:
        """Extract and sanitize PDF metadata from a PyMuPDF document."""
        table = getattr(document, "metadata", {}) or {}
        fields = (
            "author", "title", "subject", "keywords", "creator",
            "producer", "creationDate", "modDate",
        )
        metadata: Dict[str, Any] = {}
        for field in fields:
            value = table.get(field)
            if value is not None and str(value).strip():
                metadata[field] = (
                    self._parse_pdf_date(value)
                    if field in {"creationDate", "modDate"}
                    else str(value)
                )
            else:
                metadata[field] = None
        return metadata

    @staticmethod
    def _parse_pdf_date(value: str) -> Optional[str]:
        """Convert a PDF ``D:YYYYMMDDHHmmSS...`` date into ISO-8601 text."""
        if not value:
            return None
        match = _PDF_DATE_PATTERN.search(str(value))
        if not match:
            return str(value)
        try:
            parts = match.groupdict()
            year = int(parts.get("year") or 0)
            month = int(parts.get("month") or 1)
            day = int(parts.get("day") or 1)
            hour = int(parts.get("hour") or 0)
            minute = int(parts.get("minute") or 0)
            second = int(parts.get("second") or 0)
            return datetime(
                year, month, day, hour, minute, second
            ).isoformat()
        except (ValueError, TypeError):
            return str(value)

    # ------------------------------------------------------------------ #
    # DOCX
    # ------------------------------------------------------------------ #
    def _process_docx(
        self, file_path: Path, file_type: str, start: float
    ) -> Dict[str, Any]:
        """Extract paragraphs, tables, headers/footers and core properties."""
        if docx is None:
            return self._error_result(
                file_path, file_type, "python-docx is not installed", start
            )

        try:
            document = docx.Document(str(file_path))
        except Exception as exc:
            logger.warning("Could not open DOCX %s: %s", file_path, exc)
            return self._error_result(
                file_path,
                file_type,
                "Corrupted or unreadable DOCX: {}".format(exc),
                start,
            )

        try:
            paragraphs = "\n".join(
                paragraph.text.strip()
                for paragraph in document.paragraphs
                if paragraph.text and paragraph.text.strip()
            )

            tables = document.tables
            table_details = {
                "table_count": len(tables),
                "table_rows": sum(len(table.rows) for table in tables),
                "table_cells": sum(
                    len(table.rows) * len(table.columns) for table in tables
                ),
            }

            header_text, footer_text = self._extract_headers_footers(document)
            if header_text:
                paragraphs = "{}\n\n[HEADER]\n{}".format(paragraphs, header_text)
            if footer_text:
                paragraphs = "{}\n\n[FOOTER]\n{}".format(paragraphs, footer_text)

            metadata = self._extract_docx_properties(document)
            metadata.update(table_details)
            metadata["has_headers"] = bool(header_text)
            metadata["has_footers"] = bool(footer_text)
            metadata["creation_date"] = metadata.get("created")
            metadata["modification_date"] = metadata.get("modified")

            font_sizes: list = []
            alignment_values: list = []
            try:
                for paragraph in document.paragraphs:
                    alignment = paragraph.alignment
                    if alignment is not None:
                        alignment_values.append(float(int(alignment)))
                    for run in paragraph.runs:
                        size = getattr(run.font, "size", None)
                        if size is not None and getattr(size, "pt", None):
                            font_sizes.append(float(size.pt))
            except Exception:
                logger.debug("Could not collect DOCX layout stats", exc_info=True)

            image_count = 0
            try:
                image_count = len(document.inline_shapes)
            except Exception:
                pass

            ocr_confidence = 0.0
            return self._finalize(
                file_path=file_path,
                file_type=file_type,
                text=paragraphs,
                pages=1,
                metadata=metadata,
                tables_detected=len(tables),
                ocr_used=False,
                ocr_confidence=ocr_confidence,
                ocr_pages=0,
                image_only_pages=0,
                start=start,
                image_count=image_count,
                font_size_variance=_variance(font_sizes),
                alignment_variance=_variance(alignment_values),
            )
        except Exception as exc:
            logger.exception("Failed to extract text from DOCX %s", file_path)
            return self._error_result(
                file_path,
                file_type,
                "Failed to process DOCX: {}".format(exc),
                start,
            )

    def _extract_headers_footers(self, document: Any) -> Tuple[str, str]:
        """Collect header and footer text across all document sections."""
        header_parts = []
        footer_parts = []
        for section in document.sections:
            for target, collector in (
                (section.header, header_parts),
                (section.footer, footer_parts),
            ):
                try:
                    if target is None or target.is_linked_to_previous:
                        continue
                    value = (target.text or "").strip()
                    if value:
                        collector.append(value)
                except Exception:
                    logger.debug(
                        "Failed to read header/footer of a section",
                        exc_info=True,
                    )
        return "\n".join(header_parts), "\n".join(footer_parts)

    def _extract_docx_properties(self, document: Any) -> Dict[str, Any]:
        """Extract core document properties, tolerating missing fields."""
        metadata: Dict[str, Any] = {}
        properties = document.core_properties
        text_fields = (
            "author", "last_modified_by", "title", "subject", "keywords",
            "comments", "category", "content_status", "identifier",
            "language", "version", "revision",
        )
        for field in text_fields:
            metadata[field] = self._safe_property(properties, field)

        for field in ("created", "modified"):
            value = self._safe_property(properties, field)
            metadata[field] = (
                value.isoformat() if isinstance(value, datetime) else value
            )
        return metadata

    @staticmethod
    def _safe_property(obj: Any, name: str, default: Any = None) -> Any:
        """Read an attribute defensively, returning ``default`` on error."""
        try:
            return getattr(obj, name, default)
        except Exception:
            return default

    # ------------------------------------------------------------------ #
    # TXT
    # ------------------------------------------------------------------ #
    def _process_txt(
        self, file_path: Path, file_type: str, start: float
    ) -> Dict[str, Any]:
        """Read and parse a plain text file with encoding detection."""
        try:
            raw = file_path.read_bytes()
        except Exception as exc:
            logger.warning("Could not read text file %s: %s", file_path, exc)
            return self._error_result(
                file_path,
                file_type,
                "Unreadable text file: {}".format(exc),
                start,
            )

        text, encoding, has_bom = self._decode_bytes(raw)
        if b"\x00" in raw[:4096] or (
            not text.strip() and _CONTROL_CHARACTERS.search(text[:512])
        ):
            return self._error_result(
                file_path,
                file_type,
                "File appears to be binary, not plain text",
                start,
            )

        metadata = {
            "encoding": encoding,
            "has_bom": has_bom,
            "line_count": text.count("\n") + 1 if text.strip() else 0,
        }
        return self._finalize(
            file_path=file_path,
            file_type=file_type,
            text=text,
            pages=1,
            metadata=metadata,
            tables_detected=0,
            ocr_used=False,
            ocr_confidence=0.0,
            ocr_pages=0,
            image_only_pages=0,
            start=start,
        )

    @staticmethod
    def _decode_bytes(raw: bytes) -> Tuple[str, str, bool]:
        """Decode raw text bytes, trying UTF-8 then falling back to Latin-1."""
        has_bom = bool(raw) and raw.startswith(
            (b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff")
        )
        if raw.startswith(b"\xef\xbb\xbf"):
            return raw.decode("utf-8-sig"), "utf-8-sig (BOM)", True
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                return raw.decode(encoding), encoding, has_bom
            except (UnicodeDecodeError, LookupError):
                continue
        return raw.decode("latin-1", errors="replace"), "latin-1", has_bom

    # ------------------------------------------------------------------ #
    # Images
    # ------------------------------------------------------------------ #
    def _process_image(
        self, file_path: Path, file_type: str, start: float
    ) -> Dict[str, Any]:
        """OCR an image file, optionally processing every TIFF frame."""
        if PILImage is None or np is None:
            return self._error_result(
                file_path, file_type, "Pillow is not installed", start
            )

        try:
            ocr_available = self.ocr_engine.is_available()
            text_parts = []
            confidence_parts = []
            frames = 0

            with PILImage.open(str(file_path)) as image:
                base_width, base_height = image.size
                base_mode = image.mode
                while True:
                    frames += 1
                    frame = image.copy()
                    ocr_text, confidence = self._ocr_frame(frame)
                    if ocr_text.strip():
                        text_parts.append(ocr_text)
                    if confidence > 0.0:
                        confidence_parts.append(confidence)
                    try:
                        image.seek(image.tell() + 1)
                    except EOFError:
                        break

            text = "\n\n".join(text_parts)
            confidence = (
                round(float(np.mean(confidence_parts)), 2)
                if confidence_parts
                else 0.0
            )
            metadata = {
                "width": base_width,
                "height": base_height,
                "mode": base_mode,
                "frames": frames,
                "ocr_available": ocr_available,
            }
            ocr_pages = sum(1 for part in text_parts if part.strip())
            return self._finalize(
                file_path=file_path,
                file_type=file_type,
                text=text,
                pages=1,
                metadata=metadata,
                tables_detected=0,
                ocr_used=ocr_pages > 0,
                ocr_confidence=confidence,
                ocr_pages=ocr_pages,
                image_only_pages=frames,
                start=start,
            )
        except Exception as exc:
            logger.exception("Failed to OCR image %s", file_path)
            return self._error_result(
                file_path,
                file_type,
                "Failed to process image: {}".format(exc),
                start,
            )

    # ------------------------------------------------------------------ #
    # Unknown types
    # ------------------------------------------------------------------ #
    def _process_unknown(
        self, file_path: Path, file_type: str, start: float
    ) -> Dict[str, Any]:
        """Best-effort handling for unrecognized file types (try as text)."""
        try:
            raw = file_path.read_bytes()
        except Exception as exc:
            logger.warning("Could not read %s: %s", file_path, exc)
            return self._error_result(
                file_path, file_type, "Unreadable file: {}".format(exc), start
            )

        text, encoding, has_bom = self._decode_bytes(raw)
        sample = text[:512]
        if b"\x00" in raw[:4096]:
            return self._error_result(
                file_path,
                file_type,
                "Unsupported file type: {}".format(file_type),
                start,
            )
        if not text.strip() or _CONTROL_CHARACTERS.search(sample):
            return self._error_result(
                file_path,
                file_type,
                "File appears to be binary or cannot be read as text",
                start,
            )

        metadata = {
            "encoding": encoding,
            "has_bom": has_bom,
            "inferred_type": "text",
        }
        return self._finalize(
            file_path=file_path,
            file_type=file_type,
            text=text,
            pages=1,
            metadata=metadata,
            tables_detected=0,
            ocr_used=False,
            ocr_confidence=0.0,
            ocr_pages=0,
            image_only_pages=0,
            start=start,
        )

    # ------------------------------------------------------------------ #
    # Result assembly
    # ------------------------------------------------------------------ #
    def _finalize(
        self,
        file_path: Path,
        file_type: str,
        text: str,
        pages: int,
        metadata: Dict[str, Any],
        tables_detected: int,
        ocr_used: bool,
        ocr_confidence: float,
        ocr_pages: int,
        image_only_pages: int,
        start: float,
        blank_page_count: int = 0,
        image_count: int = 0,
        font_size_variance: float = 0.0,
        alignment_variance: float = 0.0,
    ) -> Dict[str, Any]:
        """Assemble the normalized result schema with statistics.

        Statistics are emitted under both their natural names (``pages``,
        ``tables_detected``) and the canonical names consumed by the feature
        engineering / marker detection modules (``page_count``,
        ``table_count``, ``font_size_variance``, ...).
        """
        stats = {
            "filename": file_path.name,
            "file_type": file_type,
            "file_size": self._file_size(file_path),
            "pages": pages,
            "page_count": int(pages),
            "words": self._word_count(text),
            "characters": len(text),
            "tables_detected": int(tables_detected),
            "table_count": int(tables_detected),
            "blank_page_count": int(blank_page_count),
            "font_size_variance": float(font_size_variance),
            "alignment_variance": float(alignment_variance),
            "image_count": int(image_count),
            "suspicious_region_count": 0,
            "total_inconsistencies": 0,
            "id_inconsistencies": 0,
            "ocr_used": bool(ocr_used),
            "ocr_pages": int(ocr_pages),
            "image_only_pages": int(image_only_pages),
            "ocr_confidence": float(round(ocr_confidence, 2)),
            "metadata_available": self._metadata_available(metadata),
            "processing_time_ms": int((time.time() - start) * 1000),
        }
        return {
            "success": True,
            "error": None,
            "text": text,
            "pages": int(pages),
            "metadata": metadata or {},
            "stats": stats,
            "ocr_used": bool(ocr_used),
        }

    def _error_result(
        self,
        file_path: Path,
        file_type: str,
        message: str,
        start: float,
    ) -> Dict[str, Any]:
        """Build a consistent failure result."""
        logger.error("Document processing failed for %s: %s", file_path, message)
        stats = {
            "filename": file_path.name,
            "file_type": file_type,
            "file_size": self._file_size(file_path),
            "pages": 0,
            "page_count": 0,
            "words": 0,
            "characters": 0,
            "tables_detected": 0,
            "table_count": 0,
            "blank_page_count": 0,
            "font_size_variance": 0.0,
            "alignment_variance": 0.0,
            "image_count": 0,
            "suspicious_region_count": 0,
            "total_inconsistencies": 0,
            "id_inconsistencies": 0,
            "ocr_used": False,
            "ocr_pages": 0,
            "image_only_pages": 0,
            "ocr_confidence": 0.0,
            "metadata_available": False,
            "processing_time_ms": int((time.time() - start) * 1000),
        }
        return {
            "success": False,
            "error": message,
            "text": "",
            "pages": 0,
            "metadata": {},
            "stats": stats,
            "ocr_used": False,
        }

    @staticmethod
    def _file_size(file_path: Path) -> int:
        """Return a file's size in bytes, or ``0`` when it cannot be read."""
        try:
            return file_path.stat().st_size
        except OSError:
            return 0

    @staticmethod
    def _word_count(text: str) -> int:
        """Count whitespace-delimited words in extracted text."""
        return len(text.split())

    @staticmethod
    def _metadata_available(metadata: Dict[str, Any]) -> bool:
        """Return whether any metadata field carries a meaningful value."""
        return any(
            value is not None and str(value).strip() != "" and value != 0
            for value in metadata.values()
        )


PDF_TYPE = "pdf"
DOCX_TYPE = "docx"
TEXT_TYPE = "txt"
IMAGE_TYPE = "image"