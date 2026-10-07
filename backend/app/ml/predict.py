"""Inference helpers for trained DocuGuard models.

The :class:`ModelPredictor` loads every artifact persisted by
:func:`app.ml.train.train_and_save_models` (the classifiers, plus the optional
scaler, TF-IDF vectorizer and feature columns) and turns a raw feature vector
into per-model genuine/fake probabilities.

Probabilities follow the project convention:

* ``genuine_prob`` - probability that the document is *original* (class ``0``).
* ``fake_prob``    - probability that the document is *forged* (class ``1``).

Missing model files are handled gracefully: the predictor simply skips the
unavailable model so the application keeps working before training has run.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
from scipy import sparse as sp

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ModelPredictor:
    """Loads persisted models and produces per-model predictions.

    Attributes:
        models: Dict mapping model name to the loaded fitted estimator.
        model_paths: Dict mapping model name to its on-disk path.
        scaler: Optional loaded :class:`StandardScaler` used to normalize input
            feature vectors before prediction.
        vectorizer: Optional loaded TF-IDF vectorizer (used by convenience
            methods that consume raw text).
    """

    # Model name to config-path attribute mapping.
    _MODEL_PATH_ATTRS: Dict[str, str] = {
        "logistic_regression": "lr_model_path",
        "random_forest": "rf_model_path",
        "xgboost": "xgb_model_path",
    }

    def __init__(self, settings: Any = None) -> None:
        self._settings = get_settings() if settings is None else settings
        self.models: Dict[str, Any] = {}
        self.model_paths: Dict[str, str] = {}
        self.scaler: Optional[Any] = None
        self.vectorizer: Optional[Any] = None
        self.feature_columns: Optional[List[str]] = None
        self._load_artifacts()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _load_artifacts(self) -> None:
        """Load every available artifact the predictor needs at runtime."""
        settings = self._settings

        for name, attr in self._MODEL_PATH_ATTRS.items():
            path = getattr(settings, attr, None)
            if not path:
                logger.warning("No configured path for model %r; skipping", name)
                continue
            self.model_paths[name] = str(path)
            model = self._safe_load(str(path), name)
            if model is not None:
                self.models[name] = model

        self.scaler = self._safe_load(settings.scaler_path, "scaler")
        self.vectorizer = self._safe_load(
            str(settings.model_dir_path / "tfidf_vectorizer.joblib"), "vectorizer"
        )
        self.feature_columns = self._safe_load(
            settings.feature_columns_path, "feature_columns"
        )

        if not self.models:
            logger.warning(
                "No trained models found in %s; train models before predicting",
                settings.model_dir_path,
            )
        else:
            logger.info(
                "ModelPredictor ready with %d model(s): %s",
                len(self.models),
                ", ".join(self.models),
            )

    def _safe_load(self, path: str, name: str) -> Any:
        """Load a ``joblib`` artifact, returning ``None`` when unavailable."""
        target = Path(path)
        if not target.is_file():
            logger.warning("%s artifact not found at %s; skipping", name, target)
            return None
        try:
            obj = joblib.load(str(target))
            logger.info("Loaded %s from %s", name, target)
            return obj
        except Exception:
            logger.exception("Failed to load %s from %s", name, target)
            return None

    # ------------------------------------------------------------------ #
    # Status helpers
    # ------------------------------------------------------------------ #
    @property
    def available_models(self) -> List[str]:
        """Names of the models loaded successfully."""
        return list(self.models.keys())

    @property
    def is_available(self) -> bool:
        """Whether at least one model could be loaded."""
        return bool(self.models)

    @property
    def missing_models(self) -> List[str]:
        """Model names with a configured path but no loaded estimator."""
        return [name for name in self.model_paths if name not in self.models]

    def artifacts_signature(self) -> float:
        """Latest mtime across the configured artifact files (``0.0`` if none).

        Used by callers to detect that a training run replaced the persisted
        models so the predictor can reload without restarting the service.
        """
        newest = 0.0
        paths = list(self.model_paths.values())
        paths.extend(
            [
                self._settings.scaler_path,
                self._settings.feature_columns_path,
                str(self._settings.model_dir_path / "tfidf_vectorizer.joblib"),
            ]
        )
        for path in paths:
            try:
                newest = max(newest, Path(path).stat().st_mtime)
            except (OSError, TypeError, ValueError):
                continue
        return newest

    def reload(self) -> bool:
        """Drop every loaded artifact and load whatever is on disk now."""
        self.models.clear()
        self.scaler = None
        self.vectorizer = None
        self.feature_columns = None
        self._load_artifacts()
        return self.is_available

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(self, feature_vector: Sequence[Any]) -> Dict[str, Dict[str, float]]:
        """Produce genuine/fake probabilities from every loaded model.

        Args:
            feature_vector: A 1-D feature vector (or batchable 2-D matrix)
                matching the feature order used during training.  Must include
                the combined TF-IDF + numeric features (or already be ordered
                for the trained scaler).

        Returns:
            Dict mapping model name to ``{"genuine_prob": float,
            "fake_prob": float}``.  Models that could not produce a prediction
            are omitted.
        """
        if not self.models:
            logger.warning("predict() called but no models are available")
            return {}

        matrix = self._to_matrix(feature_vector)
        if self.scaler is not None:
            try:
                matrix = self.scaler.transform(matrix)
            except Exception:
                logger.exception(
                    "Scaler transform failed; predicting with raw features"
                )

        results: Dict[str, Dict[str, float]] = {}
        for name, model in self.models.items():
            proba = self._predict_proba(model, matrix)
            if proba is None:
                logger.warning("No probability source for model %r; skipping", name)
                continue
            genuine_prob, fake_prob = self._split_probabilities(proba)
            results[name] = {
                "genuine_prob": round(float(genuine_prob), 6),
                "fake_prob": round(float(fake_prob), 6),
            }
            logger.debug(
                "%s -> genuine=%.4f fake=%.4f", name, genuine_prob, fake_prob
            )
        return results

    def predict_from_text_and_features(
        self,
        text: str,
        feature_vector: Sequence[Any],
    ) -> Dict[str, Dict[str, float]]:
        """Prediction convenience when a TF-IDF vectorizer is available.

        Combines the TF-IDF representation of ``text`` with the numeric
        ``feature_vector`` (replicating :func:`app.ml.preprocessing.prepare_features`
        for a single sample) and predicts from the combined vector.

        Args:
            text: Raw document text.
            feature_vector: Numeric features only (the output of the
                :class:`~app.core.feature_engineering.FeatureEngineer` dict in
                canonical order).

        Returns:
            Same structure as :func:`predict`.
        """
        if self.vectorizer is None:
            raise RuntimeError(
                "predict_from_text_and_features requires a TF-IDF vectorizer; "
                "train and persist the models first"
            )
        from app.ml.preprocessing import clean_text, combine_features  # local import

        tfidf_row = self.vectorizer.transform([clean_text(text)])
        numeric_row = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
        combined = combine_features(tfidf_row, numeric_row)
        return self.predict(combined)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_matrix(feature_vector: Any) -> Union[np.ndarray, sp.spmatrix]:
        """Coerce an input vector to a 2-D float64 matrix or sparse matrix.

        Accepts dense sequences/arrays as well as scipy sparse rows so that a
        single combined feature vector (sparse TF-IDF + numeric) can be passed
        directly.

        Args:
            feature_vector: 1-D dense vector, 2-D dense matrix, or a scipy
                sparse matrix/row.

        Returns:
            A 2-D matrix ready for the scaler and the estimators.
        """
        if sp.issparse(feature_vector):
            matrix = feature_vector.tocsr().astype(np.float64)
            if matrix.ndim != 2 or matrix.shape[0] != 1:
                raise ValueError(
                    "sparse feature_vector must be a single row "
                    "(shape (1, n_features))"
                )
            if matrix.shape[1] == 0 or np.any(~np.isfinite(matrix.data)):
                raise ValueError("feature_vector contains NaN or infinite values")
            return matrix

        matrix = np.asarray(feature_vector, dtype=np.float64)
        if matrix.ndim == 0:
            raise ValueError("feature_vector must be at least one-dimensional")
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if np.any(~np.isfinite(matrix)):
            raise ValueError("feature_vector contains NaN or infinite values")
        return matrix

    def _predict_proba(self, model: Any, X: np.ndarray) -> Optional[np.ndarray]:
        """Return per-sample probabilities, with a decision-function fallback."""
        if hasattr(model, "predict_proba"):
            try:
                return np.asarray(model.predict_proba(X))
            except Exception:
                logger.debug(
                    "predict_proba failed for %s",
                    model.__class__.__name__,
                    exc_info=True,
                )
        if hasattr(model, "decision_function"):
            try:
                scores = np.asarray(model.decision_function(X))
                return 1.0 / (1.0 + np.exp(-scores))
            except Exception:
                logger.debug(
                    "decision_function failed for %s",
                    model.__class__.__name__,
                    exc_info=True,
                )
        return None

    @staticmethod
    def _split_probabilities(proba: np.ndarray) -> tuple:
        """Split a probability array into ``(genuine, fake)`` probabilities.

        Binary classifiers report two columns in training order (``[0]`` =
        genuine, ``[1]`` = fake).  Single-column outputs (e.g. sigmoid-transformed
        decision scores) are interpreted as the fake-class probability.

        Args:
            proba: Per-sample probability array.

        Returns:
            ``(genuine_prob, fake_prob)`` for the first sample.
        """
        if proba.ndim == 2:
            sample = proba[0]
            if sample.shape[0] >= 2:
                return float(sample[0]), float(sample[1])
            fake_prob = float(sample[0])
            return 1.0 - fake_prob, fake_prob
        if proba.shape[0] == 1:
            fake_prob = float(proba[0])
            return 1.0 - fake_prob, fake_prob
        fake_prob = float(proba[1])
        return float(proba[0]), fake_prob


def load_predictor(settings: Any = None) -> ModelPredictor:
    """Convenience factory that builds and returns a :class:`ModelPredictor`."""
    return ModelPredictor(settings=settings)