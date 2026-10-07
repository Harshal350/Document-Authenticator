"""Preprocessing helpers for the DocuGuard machine-learning pipeline.

This module turns raw extracted document text and the per-document feature
dictionaries produced by :class:`~app.core.feature_engineering.FeatureEngineer`
into matrices that scikit-learn estimators can train on directly:

* :func:`clean_text`            - normalizes raw/OCR text for vectorization.
* :func:`create_tfidf_vectorizer` - builds a configured ``TfidfVectorizer``.
* :func:`build_feature_matrix`  - assembles the dense numeric feature matrix.
* :func:`combine_features`      - merges the sparse TF-IDF matrix with the
                                  numeric feature matrix into one sparse matrix.
* :func:`prepare_features`      - end-to-end helper producing all training
                                  matrices plus the label vector.
* :class:`TextPreprocessor`     - object-oriented wrapper around the above.
"""

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import get_settings
from app.core.feature_engineering import FeatureEngineer
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Canonical order of the numeric features emitted by the FeatureEngineer.
# Preserving this order keeps the feature matrix aligned across training and
# inference even when individual feature dictionaries arrive unordered.
NUMERIC_FEATURE_COLUMNS: List[str] = list(FeatureEngineer.FEATURE_NAMES)

# Compiled once so repeated cleaning calls stay fast.
_SPECIAL_CHARACTERS_PATTERN: re.Pattern = re.compile(r"[^a-z0-9\s]+")
_WHITESPACE_PATTERN: re.Pattern = re.compile(r"\s+")


def _as_float(value: Any) -> float:
    """Coerce ``value`` to a finite float, silently falling back to ``0.0``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def clean_text(text: str) -> str:
    """Normalize document text for TF-IDF vectorization.

    Lowercases the input, strips any character that is not a letter, digit or
    whitespace, collapses runs of whitespace into a single space and trims the
    result.  Used both at training time and during inference so that the
    vectorizer always sees identically shaped input.

    Args:
        text: Raw extracted or OCR document text.

    Returns:
        Cleaned, whitespace-normalized lowercase text.
    """
    if not text:
        return ""
    normalized = text.lower()
    normalized = _SPECIAL_CHARACTERS_PATTERN.sub(" ", normalized)
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()


def create_tfidf_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """Create a ``TfidfVectorizer`` configured for document forensics.

    Args:
        max_features: Cap on the vocabulary size (largest term frequencies by
            document frequency).

    Returns:
        A fitted-but-untrained ``TfidfVectorizer``.
    """
    return TfidfVectorizer(
        max_features=max(max_features, 1),
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_df=0.95,
        dtype=np.float32,
    )


def build_feature_matrix(
    feature_dicts: List[Dict[str, Any]],
    columns: Optional[List[str]] = None,
) -> np.ndarray:
    """Convert a list of feature dictionaries into a dense numeric matrix.

    Rows are aligned to the canonical :data:`NUMERIC_FEATURE_COLUMNS` order (or
    the supplied ``columns``).  Missing or non-finite values are replaced with
    ``0.0`` so downstream estimators never see ``NaN`` values.

    Args:
        feature_dicts: One feature dictionary per document.
        columns: Ordered column names; defaults to the FeatureEngineer order.

    Returns:
        Float64 array of shape ``(n_documents, n_columns)``.
    """
    column_names = list(columns) if columns is not None else NUMERIC_FEATURE_COLUMNS
    rows = [
        [_as_float(row.get(name, 0.0)) for name in column_names]
        for row in feature_dicts
    ]
    matrix = np.asarray(rows, dtype=np.float64)
    logger.debug("Built numeric feature matrix of shape %s", matrix.shape)
    return matrix


def combine_features(
    tfidf_matrix: Union[sp.spmatrix, np.ndarray],
    feature_matrix: Union[sp.spmatrix, np.ndarray],
) -> sp.csr_matrix:
    """Horizontally stack the TF-IDF and numeric feature matrices.

    The TF-IDF matrix is sparse while the numeric matrix is dense, so both are
    converted to CSR and stacked into a single CSR matrix.

    Args:
        tfidf_matrix: Sparse (or dense) TF-IDF representation of the text.
        feature_matrix: Dense (or sparse) numeric feature representation.

    Returns:
        A combined sparse CSR matrix with ``n_tfidf + n_numeric`` columns.
    """
    tfidf_csr = sp.csr_matrix(tfidf_matrix, dtype=np.float64)
    numeric_csr = sp.csr_matrix(feature_matrix, dtype=np.float64)
    if tfidf_csr.shape[0] != numeric_csr.shape[0]:
        raise ValueError(
            "tfidf_matrix and feature_matrix must have the same number of rows; "
            f"got {tfidf_csr.shape[0]} vs {numeric_csr.shape[0]}"
        )
    combined = sp.hstack([tfidf_csr, numeric_csr], format="csr", dtype=np.float64)
    logger.debug("Combined feature matrix shape: %s", combined.shape)
    return combined


def build_feature_columns(
    vectorizer: TfidfVectorizer,
    numeric_columns: Optional[List[str]] = None,
) -> List[str]:
    """Return the full ordered list of column names for a combined matrix.

    TF-IDF vocabulary names come first (in vectorizer vocabulary order),
    followed by the numeric feature names.

    Args:
        vectorizer: A trained :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.
        numeric_columns: Numeric feature names; defaults to the canonical order.

    Returns:
        Combined list of feature column names.
    """
    numeric_names = (
        list(numeric_columns) if numeric_columns is not None else NUMERIC_FEATURE_COLUMNS
    )
    if not hasattr(vectorizer, "vocabulary_"):
        raise ValueError(
            "vectorizer is not fitted; call fit/fit_transform before "
            "requesting feature columns"
        )
    tfidf_names = [str(name) for name in vectorizer.get_feature_names_out()]
    return tfidf_names + numeric_names


def prepare_features(
    feature_dicts: List[Dict[str, Any]],
    text_list: List[str],
    labels: Optional[Sequence[Any]] = None,
) -> Tuple[sp.csr_matrix, np.ndarray, sp.csr_matrix, Optional[np.ndarray]]:
    """Build every matrix the ML pipeline needs from raw inputs.

    Args:
        feature_dicts: Per-document feature dictionaries (one per sample).
        text_list: Raw document text, one string per sample.
        labels: Optional ground-truth labels aligned with the samples
            (``0`` = genuine, ``1`` = fake).

    Returns:
        A tuple ``(X_text_tfidf, X_features, combined_X, y)`` where:

        * ``X_text_tfidf``  - sparse TF-IDF matrix over the cleaned text,
        * ``X_features``    - dense numeric feature matrix,
        * ``combined_X``    - TF-IDF and numeric features stacked together,
        * ``y``             - label vector (``None`` when labels are omitted).
    """
    n_samples = len(text_list)
    if len(feature_dicts) != n_samples:
        raise ValueError(
            "feature_dicts and text_list must have the same length; "
            f"got {len(feature_dicts)} vs {n_samples}"
        )
    if labels is not None and len(labels) != n_samples:
        raise ValueError(
            "labels must have the same length as text_list; "
            f"got {len(labels)} vs {n_samples}"
        )

    cleaned_texts = [clean_text(text) for text in text_list]
    vectorizer = create_tfidf_vectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    feature_matrix = build_feature_matrix(feature_dicts)
    combined_matrix = combine_features(tfidf_matrix, feature_matrix)

    y: Optional[np.ndarray] = None
    if labels is not None:
        y = np.asarray(labels)

    logger.info(
        "Prepared features: text=%s numeric=%s combined=%s samples=%d",
        tfidf_matrix.shape,
        feature_matrix.shape,
        combined_matrix.shape,
        n_samples,
    )
    return tfidf_matrix, feature_matrix, combined_matrix, y


class TextPreprocessor:
    """Object-oriented wrapper around the text cleaning / vectorization helpers.

    The preprocessor owns a :class:`~sklearn.feature_extraction.text.TfidfVectorizer`
    and exposes its own ``fit`` / ``transform`` / ``fit_transform`` interface so
    it can be persisted and reused for later inference.
    """

    def __init__(self, max_features: int = 5000) -> None:
        self.max_features: int = max_features
        self.vectorizer: TfidfVectorizer = create_tfidf_vectorizer(max_features)

    def clean_text(self, text: str) -> str:
        """Clean a single text string (see :func:`clean_text`)."""
        return clean_text(text)

    def _cleaned(self, texts: Sequence[str]) -> List[str]:
        return [clean_text(text) for text in texts]

    def fit(self, texts: Sequence[str]) -> "TextPreprocessor":
        """Fit the internal TF-IDF vectorizer on cleaned texts."""
        self.vectorizer.fit(self._cleaned(texts))
        logger.info(
            "Fitted TF-IDF vectorizer on %d documents (%d terms)",
            len(texts),
            len(self.vectorizer.vocabulary_),
        )
        return self

    def transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        """Transform cleaned texts into a TF-IDF matrix."""
        return self.vectorizer.transform(self._cleaned(texts))

    def fit_transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        """Fit then transform; returns the TF-IDF matrix for the inputs."""
        return self.vectorizer.fit_transform(self._cleaned(texts))

    def feature_names(self, numeric_columns: Optional[List[str]] = None) -> List[str]:
        """Return combined feature column names for this preprocessor."""
        return build_feature_columns(self.vectorizer, numeric_columns)