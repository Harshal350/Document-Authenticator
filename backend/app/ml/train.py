"""Model training routines for the DocuGuard fake-document classification.

Trains three complementary binary classifiers:

* :class:`~sklearn.linear_model.LogisticRegression` - fast, interpretable linear
  baseline that makes the text/numeric boundary explicit.
* :class:`~sklearn.ensemble.RandomForestClassifier` - robust non-linear model
  that handles mixed dense/sparse features well.
* XGBoost - the strongest non-linear classifier when the runtime is available.
  On machines where the ``xgboost`` library cannot load (e.g. missing OpenMP on
  macOS) the pipeline transparently falls back to a scikit-learn
  :class:`~sklearn.ensemble.GradientBoostingClassifier` so training never
  breaks.

All trained artifacts (classifiers, scaler, feature columns and the optional
TF-IDF vectorizer) are persisted with ``joblib`` to the paths exposed by
:func:`app.config.get_settings`.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Random state used everywhere so runs are reproducible.
_RANDOM_STATE: int = 42

# Canonical model names returned by :func:`train_all_models`.  The values pair
# with the config paths used when persisting each artifact.
DEFAULT_MODEL_NAMES: List[str] = [
    "logistic_regression",
    "random_forest",
    "xgboost",
]


# --------------------------------------------------------------------------- #
# Estimator builders
# --------------------------------------------------------------------------- #
def _logistic_regression() -> LogisticRegression:
    """Build the logistic regression classifier for the pipeline."""
    return LogisticRegression(
        max_iter=2000,
        C=0.5,
        solver="liblinear",
        class_weight="balanced",
        random_state=_RANDOM_STATE,
    )


def _random_forest() -> RandomForestClassifier:
    """Build the random forest classifier for the pipeline."""
    return RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=_RANDOM_STATE,
    )


def _gradient_boosting() -> GradientBoostingClassifier:
    """Build the scikit-learn gradient boosting fallback classifier."""
    return GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=2,
        subsample=0.8,
        max_features="sqrt",
        random_state=_RANDOM_STATE,
    )


def _build_xgboost() -> Any:
    """Build an XGBoost classifier, falling back to gradient boosting.

    The ``xgboost`` package sometimes fails to initialize because the native
    OpenMP runtime (``libomp``) is missing - most commonly on macOS.  Rather
    than abort the whole training run we silently substitute
    :class:`~sklearn.ensemble.GradientBoostingClassifier`, which produces a
    comparable model quality for this task.

    Returns:
        An ``XGBClassifier`` when available, otherwise a ``GradientBoostingClassifier``.
    """
    try:
        from xgboost import XGBClassifier  # noqa: PLC0415 - optional dependency
    except Exception as exc:  # pragma: no cover - environment specific
        logger.warning(
            "XGBoost unavailable (%s); falling back to GradientBoostingClassifier",
            exc,
        )
        return _gradient_boosting()

    try:
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=_RANDOM_STATE,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Could not construct XGBClassifier (%s); using gradient boosting fallback",
            exc,
        )
        return _gradient_boosting()


def _fit_model(model: Any, X_train: Any, y_train: Sequence[Any]) -> Any:
    """Fit ``model`` and record its wall-clock training time as an attribute.

    The elapsed seconds are stored as ``training_time_seconds_`` on the fitted
    estimator so evaluation / persistence steps can report them later.

    Args:
        model: Un-fitted estimator.
        X_train: Training matrix (dense or sparse).
        y_train: Ground-truth labels aligned with the rows of ``X_train``.

    Returns:
        The fitted estimator.
    """
    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    try:
        model.training_time_seconds_ = float(elapsed)
    except Exception:  # pragma: no cover - defensive
        pass
    logger.info(
        "Trained %s on %d samples in %.2fs",
        model.__class__.__name__,
        int(X_train.shape[0]),
        elapsed,
    )
    return model


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def train_all_models(
    X_train: Any,
    y_train: Sequence[Any],
    feature_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Train the complete set of DocuGuard classifiers.

    Args:
        X_train: Feature matrix (dense :class:`numpy.ndarray` or sparse
            ``scipy.csr_matrix``) for the training split.
        y_train: Ground-truth labels (``0`` = genuine, ``1`` = fake).
        feature_columns: Optional ordered feature names used for logging and
            sanity checks against the matrix width.

    Returns:
        Dict mapping model name to a fitted estimator, always containing
        ``logistic_regression``, ``random_forest`` and ``xgboost`` keys.
    """
    X_train = X_train
    if X_train.shape[0] == 0:
        raise ValueError("train_all_models requires at least one training sample")

    if feature_columns:
        expected = len(feature_columns)
        if X_train.shape[1] != expected:
            logger.warning(
                "feature_columns length (%d) does not match X_train width (%d)",
                expected,
                X_train.shape[1],
            )
        logger.debug(
            "Training on %d features: %s",
            expected,
            ", ".join(feature_columns[:5]) + ("..." if expected > 5 else ""),
        )

    y = np.asarray(y_train)
    if len(np.unique(y)) < 2:
        logger.warning(
            "Training set contains a single class (%s); model calibration will "
            "be unreliable",
            np.unique(y).tolist(),
        )

    models: Dict[str, Any] = {
        "logistic_regression": _fit_model(_logistic_regression(), X_train, y),
        "random_forest": _fit_model(_random_forest(), X_train, y),
        "xgboost": _fit_model(_build_xgboost(), X_train, y),
    }
    logger.info(
        "Finished training %d models on %d samples / %d features",
        len(models),
        X_train.shape[0],
        X_train.shape[1],
    )
    return models


def train_and_save_models(
    X_train: Any,
    y_train: Sequence[Any],
    feature_columns: Optional[List[str]] = None,
    vectorizer: Optional[Any] = None,
    save: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Scale, train, persist and describe a full model training run.

    A :class:`~sklearn.preprocessing.StandardScaler` is fitted on the training
    matrix (sparse-safe, ``with_mean=False``) so gradient-based models behave
    well, and the same scaler is saved for use at inference time.

    Args:
        X_train: Training feature matrix (dense or sparse).
        y_train: Ground-truth labels.
        feature_columns: Optional ordered feature names.
        vectorizer: Optional fitted TF-IDF vectorizer to persist alongside the
            models so future inference can reproduce the text features.
        save: When ``True`` every artifact is written with ``joblib``.

    Returns:
        A ``(models, metadata)`` tuple; ``models`` maps model names to fitted
        estimators and ``metadata`` describes the training run (timings, sample
        counts, feature count, target classes, etc.).
    """
    settings = get_settings()
    started_at = datetime.utcnow()
    started = time.perf_counter()

    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X_train)

    models = train_all_models(X_scaled, y_train, feature_columns)
    total_elapsed = time.perf_counter() - started

    metadata: Dict[str, Any] = {
        "trained_at": started_at.isoformat() + "Z",
        "model_names": list(models.keys()),
        "training_samples": int(X_train.shape[0]),
        "feature_count": int(X_train.shape[1]),
        "feature_columns": [str(c) for c in feature_columns] if feature_columns else [],
        "target_classes": [int(c) for c in np.unique(y_train)],
        "scaler": scaler.__class__.__name__,
        "total_training_time_seconds": round(total_elapsed, 4),
        "training_time_seconds": {
            name: round(float(getattr(model, "training_time_seconds_", 0.0)), 4)
            for name, model in models.items()
        },
    }

    if save:
        persist_artifacts(models, scaler, feature_columns, vectorizer, metadata)

    return models, metadata


def persist_artifacts(
    models: Dict[str, Any],
    scaler: Optional[StandardScaler] = None,
    feature_columns: Optional[List[str]] = None,
    vectorizer: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Write the trained artifacts to disk with ``joblib``.

    Args:
        models: Dict mapping model name to a fitted estimator.
        scaler: Optional fitted scaler to persist.
        feature_columns: Optional feature name list to persist.
        vectorizer: Optional fitted TF-IDF vectorizer to persist.
        metadata: Optional training metadata persisted as a joblib artifact.

    Returns:
        Dict mapping artifact label to the path it was written to.
    """
    settings = get_settings()
    paths: Dict[str, str] = {
        "logistic_regression": settings.lr_model_path,
        "random_forest": settings.rf_model_path,
        "xgboost": settings.xgb_model_path,
    }

    written: Dict[str, Path] = {}
    for name, model in models.items():
        target = paths.get(name)
        if not target:
            logger.warning("No configured path for model %r; skipping save", name)
            continue
        written[name] = _save_joblib(model, target, "model")

    if scaler is not None:
        written["scaler"] = _save_joblib(scaler, settings.scaler_path, "scaler")

    if feature_columns is not None:
        written["feature_columns"] = _save_joblib(
            list(feature_columns), settings.feature_columns_path, "feature_columns"
        )

    if vectorizer is not None:
        vectorizer_path = str(settings.model_dir_path / "tfidf_vectorizer.joblib")
        written["vectorizer"] = _save_joblib(vectorizer, vectorizer_path, "vectorizer")

    if metadata is not None:
        metadata_path = str(settings.model_dir_path / "training_metadata.joblib")
        written["metadata"] = _save_joblib(metadata, metadata_path, "metadata")

    logger.info(
        "Persisted %d artifact(s) to %s",
        len(written),
        settings.model_dir_path,
    )
    return written


def _save_joblib(obj: Any, path: str, label: str) -> Path:
    """Persist ``obj`` to ``path``, creating the parent directory if needed."""
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, str(target))
        logger.info("Saved %s -> %s", label, target)
        return target
    except Exception:
        logger.exception("Failed to save %s -> %s", label, target)
        raise


# --------------------------------------------------------------------------- #
# End-to-end training from a labelled CSV dataset
# --------------------------------------------------------------------------- #
def load_labelled_dataset(
    dataset_path: Union[str, Path],
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
) -> Tuple["pd.DataFrame", str, str]:
    """Load a labelled document CSV and resolve its text / label columns.

    The canonical format is ``document_id,text,document_type,label`` where
    ``label = 0`` is genuine and ``label = 1`` is fake, but common aliases
    (``content``/``body``, ``target``/``class``/``is_fake``) and textual
    labels (``genuine``/``fake``) are accepted transparently.

    Args:
        dataset_path: Path to the CSV file.
        text_column: Explicit text column name (auto-detected when omitted).
        label_column: Explicit label column name (auto-detected when omitted).

    Returns:
        ``(dataframe, text_column, label_column)`` with the label column
        coerced to integers ``0`` / ``1``.

    Raises:
        ValueError: When no usable text or label column can be found.
    """
    import pandas as pd

    path = Path(dataset_path)
    if not path.is_file():
        raise ValueError(f"Dataset file not found: {path}")

    dataframe = pd.read_csv(str(path), comment="#")
    if dataframe.empty:
        raise ValueError(f"Dataset at {path} contains no rows")

    columns = {str(column).strip().lower(): column for column in dataframe.columns}

    def _resolve(explicit: Optional[str], candidates: Sequence[str]) -> str:
        if explicit:
            key = explicit.strip().lower()
            if key not in columns:
                raise ValueError(
                    f"Column '{explicit}' not found in dataset. "
                    f"Available columns: {list(dataframe.columns)}"
                )
            return columns[key]
        for candidate in candidates:
            if candidate in columns:
                return columns[candidate]
        raise ValueError(
            "Could not locate the {} column. Available columns: {}".format(
                "text" if candidates[0] == "text" else "label",
                list(dataframe.columns),
            )
        )

    text_col = _resolve(
        text_column, ("text", "document_text", "content", "body", "raw_text")
    )
    label_col = _resolve(
        label_column, ("label", "target", "is_fake", "class", "forged", "y")
    )

    dataframe = dataframe.dropna(subset=[text_col])
    dataframe[text_col] = dataframe[text_col].astype(str)

    # Normalise textual / boolean labels onto {0, 1}.
    raw_labels = dataframe[label_col]
    if raw_labels.dtype == object:
        mapping = {
            "0": 0, "1": 1,
            "genuine": 0, "original": 0, "real": 0, "authentic": 0, "legit": 0,
            "fake": 1, "forged": 1, "fraud": 1, "fraudulent": 1, "tampered": 1,
        }
        dataframe[label_col] = (
            raw_labels.astype(str).str.strip().str.lower().map(mapping)
        )
    else:
        dataframe[label_col] = raw_labels.astype(float).astype(int)

    dataframe = dataframe.dropna(subset=[label_col])
    dataframe[label_col] = dataframe[label_col].astype(int)
    if not set(dataframe[label_col].unique()).issubset({0, 1}):
        raise ValueError(
            f"Label column '{label_col}' must contain exactly two classes "
            f"(0 = genuine, 1 = fake); found "
            f"{sorted(dataframe[label_col].unique().tolist())}"
        )
    if dataframe.empty:
        raise ValueError("Dataset contains no usable labelled rows")

    logger.info(
        "Loaded dataset %s: %d rows, text=%r label=%r, class balance=%s",
        path,
        len(dataframe),
        text_col,
        label_col,
        dataframe[label_col].value_counts().to_dict(),
    )
    return dataframe, text_col, label_col


def _feature_dicts_for_dataset(
    dataframe: Any, text_col: str
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return per-row numeric feature dictionaries for a dataset.

    When the CSV already carries the canonical numeric feature columns they
    are used directly (richer dataset); otherwise the features are extracted
    from the document text with :class:`FeatureEngineer`.
    """
    from app.core.feature_engineering import FeatureEngineer
    from app.ml.preprocessing import NUMERIC_FEATURE_COLUMNS

    present = [
        column
        for column in NUMERIC_FEATURE_COLUMNS
        if column in dataframe.columns
    ]
    if len(present) == len(NUMERIC_FEATURE_COLUMNS):
        logger.info("Dataset carries all %d numeric features", len(present))
        rows = []
        for _, row in dataframe.iterrows():
            rows.append(
                {
                    column: float(row[column] or 0.0)
                    for column in NUMERIC_FEATURE_COLUMNS
                }
            )
        return rows, list(NUMERIC_FEATURE_COLUMNS)

    engineer = FeatureEngineer()
    rows = []
    texts = dataframe[text_col].tolist()
    for index, text in enumerate(texts):
        if index % 200 == 0:
            logger.info("Extracting features for %d / %d rows", index, len(texts))
        try:
            rows.append(engineer.extract_features(str(text), {}, {}))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Feature extraction failed for row %d: %s", index, exc)
            rows.append({name: 0.0 for name in NUMERIC_FEATURE_COLUMNS})
    return rows, list(NUMERIC_FEATURE_COLUMNS)


def train_from_dataset(
    dataset_path: Union[str, Path],
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    test_size: float = 0.2,
    chart_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Full training run: load data, split, featurise, train, evaluate, persist.

    This is the single entry point used by ``POST /api/v1/models/train``.
    The split is stratified (80/20 by default), the TF-IDF vectorizer and
    ``StandardScaler`` are fit on the *training* split only to avoid leakage,
    and every artifact (models, scaler, vectorizer, feature columns, metadata)
    is persisted with ``joblib``.

    Args:
        dataset_path: Labelled CSV path.
        text_column: Optional explicit text column.
        label_column: Optional explicit label column.
        test_size: Fraction of the dataset held out for evaluation.
        chart_dir: Where to render comparison charts (defaults to the
            configured ``report_dir``).

    Returns:
        ``(metrics, summary)`` where ``metrics`` is the
        :func:`app.ml.evaluate.evaluate_models` bundle and ``summary``
        describes the run (dataset size, splits, class balance, timings).
    """
    from sklearn.model_selection import train_test_split

    from app.ml.evaluate import evaluate_models, generate_comparison_charts
    from app.ml.preprocessing import (
        build_feature_columns,
        clean_text,
        combine_features,
        create_tfidf_vectorizer,
        build_feature_matrix,
    )

    started = time.perf_counter()
    settings = get_settings()
    path = Path(dataset_path)

    dataframe, text_col, label_col = load_labelled_dataset(
        path, text_column, label_column
    )
    labels = dataframe[label_col]
    class_counts = {
        "genuine": int((labels == 0).sum()),
        "fake": int((labels == 1).sum()),
    }

    indices = list(range(len(dataframe)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=_RANDOM_STATE,
        stratify=labels.tolist(),
    )
    train_frame = dataframe.iloc[train_indices]
    test_frame = dataframe.iloc[test_indices]

    # Numeric features (either supplied by a richer CSV or extracted).
    feature_dicts, numeric_columns = _feature_dicts_for_dataset(
        dataframe, text_col
    )
    train_features = [feature_dicts[index] for index in train_indices]
    test_features = [feature_dicts[index] for index in test_indices]

    # TF-IDF fit on the training split only.
    vectorizer = create_tfidf_vectorizer()
    train_texts = [clean_text(str(t)) for t in train_frame[text_col].tolist()]
    test_texts = [clean_text(str(t)) for t in test_frame[text_col].tolist()]
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_test_tfidf = vectorizer.transform(test_texts)

    X_train_numeric = build_feature_matrix(train_features, numeric_columns)
    X_test_numeric = build_feature_matrix(test_features, numeric_columns)

    X_train = combine_features(X_train_tfidf, X_train_numeric)
    X_test = combine_features(X_test_tfidf, X_test_numeric)
    y_train = np.asarray(train_frame[label_col].tolist())
    y_test = np.asarray(test_frame[label_col].tolist())

    feature_columns = build_feature_columns(vectorizer, numeric_columns)

    # Scale (sparse-safe) and train.
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = train_all_models(X_train_scaled, y_train, feature_columns)
    metrics = evaluate_models(models, X_test_scaled, y_test)

    total_elapsed = time.perf_counter() - started
    metadata: Dict[str, Any] = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "dataset_path": str(path),
        "model_names": list(models.keys()),
        "training_samples": int(len(train_indices)),
        "testing_samples": int(len(test_indices)),
        "class_distribution": class_counts,
        "feature_count": int(X_train_scaled.shape[1]),
        "feature_columns": [str(column) for column in feature_columns],
        "target_classes": [0, 1],
        "scaler": scaler.__class__.__name__,
        "test_size": float(test_size),
        "total_training_time_seconds": round(total_elapsed, 4),
        "training_time_seconds": {
            name: round(float(getattr(model, "training_time_seconds_", 0.0)), 4)
            for name, model in models.items()
        },
    }

    persist_artifacts(models, scaler, feature_columns, vectorizer, metadata)

    generated_charts: List[str] = []
    if chart_dir is None:
        chart_dir = settings.report_dir_path
    try:
        generated_charts = generate_comparison_charts(metrics, chart_dir)
    except Exception as exc:
        logger.warning("Chart generation failed: %s", exc)

    summary: Dict[str, Any] = {
        "dataset_path": str(path),
        "dataset_rows": int(len(dataframe)),
        "text_column": text_col,
        "label_column": label_col,
        "class_distribution": class_counts,
        "training_samples": int(len(train_indices)),
        "testing_samples": int(len(test_indices)),
        "feature_count": int(X_train_scaled.shape[1]),
        "model_names": list(models.keys()),
        "total_training_time_seconds": round(total_elapsed, 4),
        "training_time_seconds": metadata["training_time_seconds"],
        "charts_generated": [str(chart) for chart in generated_charts],
        "is_demo_data": path.name == "documents.csv",
    }
    logger.info(
        "Training run complete: %d train / %d test samples, features=%d, %.2fs",
        summary["training_samples"],
        summary["testing_samples"],
        summary["feature_count"],
        total_elapsed,
    )
    return metrics, summary