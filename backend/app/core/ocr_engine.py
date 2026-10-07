"""OCR engine for DocuGuard's document processing pipeline.

This module provides the :class:`OCREngine` class which wraps Tesseract OCR
(via ``pytesseract``) behind OpenCV based image preprocessing. Every external
integration point (Tesseract binary, OpenCV, PyMuPDF, Pillow) is guarded so
that the application degrades gracefully when a dependency or the Tesseract
binary is not installed on the host.
"""

import logging
import shutil
from typing import Any, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:  # pragma: no cover - environment specific
    import cv2
except Exception:
    cv2 = None
    logger.debug("OpenCV (cv2) is not available; OCR preprocessing disabled.")

try:  # pragma: no cover - environment specific
    import pytesseract
except Exception:
    pytesseract = None
    logger.debug("pytesseract is not available; OCR disabled.")

try:  # pragma: no cover - environment specific
    import fitz
except Exception:
    fitz = None
    logger.debug("PyMuPDF (fitz) is not available; PDF page OCR disabled.")

try:  # pragma: no cover - environment specific
    from PIL import Image
except Exception:
    Image = None
    logger.debug("Pillow (PIL) is not available; image conversion disabled.")

ImageLike = Union[np.ndarray, Image.Image if Image is not None else Any]


class OCREngine:
    """Performs OCR on images and scanned PDF pages.

    Handles the full OCR pipeline (render → preprocess → recognize) and
    returns both the recognized text and an aggregate confidence score.
    When the Tesseract binary is missing, all public methods return safe
    defaults (empty text, zero confidence) instead of raising.
    """

    DEFAULT_DPI = 300
    DEFAULT_PSM = 6
    DEFAULT_LANG = "eng"
    MIN_CONFIDENCE_FILTER = 0.0

    def __init__(self, lang: str = DEFAULT_LANG) -> None:
        """Initialize the engine with an OCR language.

        Args:
            lang: Tesseract language pack identifier (e.g. ``"eng"``).
        """
        self._lang = lang
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """Return whether the Tesseract binary is usable.

        The result is cached after the first check. OpenCV and pytesseract
        must also be importable for OCR to be considered available.

        Returns:
            ``True`` when OCR can be executed, ``False`` otherwise.
        """
        if pytesseract is None:
            self._available = False
        elif self._available is None:
            try:
                self._available = shutil.which("tesseract") is not None
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to probe for the tesseract binary")
                self._available = False
        return bool(self._available)

    def perform_ocr(
        self,
        image: ImageLike,
        psm: int = DEFAULT_PSM,
        lang: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Run OCR on a single image.

        The image may be a ``numpy.ndarray`` (BGR or grayscale) or a
        ``PIL.Image.Image``. It is preprocessed before recognition.

        Args:
            image: The source image to recognize.
            psm: Tesseract page segmentation mode (default ``6``).
            lang: Tesseract language override; defaults to engine setting.

        Returns:
            A tuple of ``(text, confidence)`` where ``confidence`` is the
            mean per-word confidence (0–100). Empty text and ``0.0`` are
            returned when OCR is unavailable or fails.
        """
        if not self.is_available() or cv2 is None or pytesseract is None:
            logger.warning("OCR skipped: tesseract/OpenCV not available")
            return "", 0.0

        try:
            processed = self.preprocess_image(image)
            if processed is None or processed.size == 0:
                logger.warning("OCR skipped: preprocessed image is empty")
                return "", 0.0
            if int(processed.max()) == 0:
                logger.warning("OCR skipped: preprocessed image is fully black")
                return "", 0.0

            config = "--oem 3 --psm {}".format(int(psm))
            data = pytesseract.image_to_data(
                processed,
                lang=lang or self._lang,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
            words: list = []
            confidences: list = []
            for index, raw in enumerate(data.get("text", [])):
                word = str(raw).strip()
                if not word:
                    continue
                words.append(word)
                try:
                    confidence = float(data["conf"][index])
                except (KeyError, IndexError, TypeError, ValueError):
                    confidence = -1.0
                if confidence > self.MIN_CONFIDENCE_FILTER:
                    confidences.append(confidence)

            text = " ".join(words)
            confidence = (
                round(float(np.mean(confidences)), 2) if confidences else 0.0
            )
            logger.debug(
                "OCR complete: %d words, mean confidence %.2f",
                len(words),
                confidence,
            )
            return text, confidence
        except Exception:
            logger.exception("OCR failed for the supplied image")
            return "", 0.0

    def preprocess_image(self, image: ImageLike) -> Optional[np.ndarray]:
        """Preprocess an image for OCR.

        Pipeline: convert to grayscale → bilateral denoise → Otsu binary
        threshold → deskew.

        Args:
            image: A ``numpy.ndarray`` (BGR/RGB/gray) or ``PIL.Image.Image``.

        Returns:
            A grayscale ``numpy.ndarray`` ready for Tesseract, or ``None`` on
            failure.
        """
        if cv2 is None:
            logger.warning("Image preprocessing unavailable (OpenCV missing)")
            return None
        try:
            array = self._to_ndarray(image)
            if array is None or array.size == 0:
                logger.warning("Image preprocessing skipped: empty input")
                return None

            if array.ndim == 2:
                gray = array
            elif array.ndim == 3 and array.shape[2] == 4:
                gray = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

            if int(gray.min()) >= 250:
                logger.debug("Image appears blank; skipping preprocessing")
                return gray

            denoised = cv2.bilateralFilter(
                gray, d=9, sigmaColor=80, sigmaSpace=80
            )
            _, thresholded = cv2.threshold(
                denoised, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
            return self._deskew(thresholded)
        except Exception:
            logger.exception("Image preprocessing failed")
            return None

    def handle_image_page(
        self,
        page: Any,
        dpi: int = DEFAULT_DPI,
        psm: int = DEFAULT_PSM,
    ) -> Tuple[str, float]:
        """Render a PDF page to an image and OCR it.

        Args:
            page: A PyMuPDF ``fitz.Page`` instance.
            dpi: Render resolution (dots per inch).
            psm: Tesseract page segmentation mode.

        Returns:
            A tuple of ``(text, confidence)`` for the rendered page.
        """
        if fitz is None or cv2 is None or not self.is_available():
            logger.warning("PDF page OCR skipped: dependencies unavailable")
            return "", 0.0
        try:
            pixmap = page.get_pixmap(dpi=int(dpi))
            samples = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, pixmap.n
            )
            if samples.ndim == 3 and samples.shape[2] == 4:
                gray = cv2.cvtColor(samples, cv2.COLOR_RGBA2GRAY)
            elif samples.ndim == 3 and samples.shape[2] == 3:
                gray = cv2.cvtColor(samples, cv2.COLOR_RGB2GRAY)
            else:
                gray = samples.reshape(pixmap.height, pixmap.width)

            processed = self.preprocess_image(gray)
            if processed is None:
                logger.warning("PDF page preprocessing failed")
                return "", 0.0
            return self.perform_ocr(processed, psm=psm)
        except Exception:
            logger.exception("Failed to OCR PDF page %r", page.number)
            return "", 0.0

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Rotate a binary image so text lines are roughly horizontal.

        Args:
            image: A single-channel (grayscale/binary) image.

        Returns:
            The deskewed image, or the input unchanged when the correction
            angle is negligible or the computation fails.
        """
        try:
            coords = np.column_stack(np.where(image < 128))
            if coords.shape[0] < 3:
                return image
            points = coords[:, ::-1].astype(np.float32)
            angle = cv2.minAreaRect(points)[-1]
            if angle < -45.0:
                angle = -(90.0 + angle)
            else:
                angle = -angle
            angle = max(-45.0, min(45.0, angle))
            if abs(angle) < 0.5:
                return image
            height, width = image.shape[:2]
            center = (width / 2.0, height / 2.0)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(
                image,
                matrix,
                (width, height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
        except Exception:
            logger.exception("Deskew failed; returning original image")
            return image

    def _to_ndarray(self, image: ImageLike) -> Optional[np.ndarray]:
        """Normalize a PIL image or ndarray into a BGR/grayscale ndarray.

        Args:
            image: Input image.

        Returns:
            A ``numpy.ndarray`` or ``None`` when the input is unusable.
        """
        if isinstance(image, np.ndarray):
            return image
        if Image is not None and isinstance(image, Image.Image) and cv2 is not None:
            try:
                array = np.asarray(image)
                if array.ndim == 3 and array.shape[2] == 3:
                    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                if array.ndim == 3 and array.shape[2] == 4:
                    return cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA)
                return array
            except Exception:
                logger.exception("Failed to convert PIL image to ndarray")
                return None
        logger.warning(
            "Unsupported OCR image input type: %s", type(image).__name__
        )
        return None