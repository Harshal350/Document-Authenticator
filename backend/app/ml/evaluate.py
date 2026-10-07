"""Evaluation utilities for the DocuGuard machine-learning pipeline.

Provides:

* :func:`evaluate_models` - compute per-model metrics (accuracy, precision,
  recall, F1, ROC-AUC) together with confusion matrices and ROC-curve samples.
* :func:`save_metrics_to_db` - upsert metrics into the ``model_metrics`` table.
* :func:`generate_comparison_charts` - render comparison charts (bar chart,
  ROC overlay, confusion-matrix heatmaps) with a black/green visual theme.
* :func:`evaluate_pipeline` - convenience wrapper that performs all three steps.

Labels follow the project convention: ``0`` = genuine (original), ``1`` = fake
(forged).  The positive class is therefore ``1``.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; must run before pyplot import

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from app.config import get_settings
from app.database.database import SessionLocal
from app.models.database_models import ModelMetric
from app.utils.logging import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Chart theme (black / green)
# --------------------------------------------------------------------------- #
GREEN: str = "#2ECC71"
DARK_GREEN: str = "#179A4F"
LIME_GREEN: str = "#A3E635"
PANEL_BLACK: str = "#0C0C10"
PANEL_GREY: str = "#141419"
TEXT_LIGHT: str = "#E8F5EC"
TEXT_MUTED: str = "#8FBF9F"
GRID_LINE: str = "#22332A"

# Metric keys shown in the comparison bar chart, in display order.
_COMPARISON_METRICS: List[str] = ["accuracy", "precision", "recall", "f1", "roc_auc"]
_MODEL_PALETTE: List[str] = [GREEN, LIME_GREEN, DARK_GREEN]
_METADATA_KEY: str = "metadata"


def _apply_theme() -> None:
    """Apply the black/green matplotlib theme for all generated charts."""
    plt.rcParams.update(
        {
            "figure.facecolor": PANEL_BLACK,
            "axes.facecolor": PANEL_GREY,
            "axes.edgecolor": GREEN,
            "axes.labelcolor": TEXT_LIGHT,
            "text.color": TEXT_LIGHT,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "grid.color": GRID_LINE,
            "grid.alpha": 0.4,
            "legend.facecolor": PANEL_GREY,
            "legend.edgecolor": GREEN,
            "legend.framealpha": 0.95,
            "font.size": 11,
        }
    )


# --------------------------------------------------------------------------- #
# Prediction helpers (shared with predict.py style)
# --------------------------------------------------------------------------- #
def _predict_proba(model: Any, X_test: Any) -> Optional[np.ndarray]:
    """Return per-sample probabilities, falling back to ``decision_function``.

    Returns ``None`` when the model supports neither ``predict_proba`` nor a
    usable ``decision_function``.

    Args:
        model: A fitted classifier.
        X_test: Feature matrix.

    Returns:
        Probability array (or sigmoid-transformed decision scores); ``None``
        when no probability source is available.
    """
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(X_test))
        except Exception:
            logger.debug(
                "predict_proba failed for %s",
                model.__class__.__name__,
                exc_info=True,
            )
    if hasattr(model, "decision_function"):
        try:
            scores = np.asarray(model.decision_function(X_test))
            return 1.0 / (1.0 + np.exp(-scores))
        except Exception:
            logger.debug(
                "decision_function failed for %s",
                model.__class__.__name__,
                exc_info=True,
            )
    return None


def _positive_scores(proba: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Extract the positive-class (fake) scores from a probability array."""
    if proba is None:
        return None
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1]
    if proba.ndim == 2:
        return proba[:, 0]
    return proba


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _safe_roc_auc(y_true: np.ndarray, proba_positive: Optional[np.ndarray]) -> float:
    """Compute ROC-AUC, guarding against degenerate (single-class) inputs."""
    if proba_positive is None or len(np.unique(y_true)) < 2:
        logger.warning("ROC-AUC undefined for this data (missing probabilities or single-class)")
        return 0.0
    try:
        return float(roc_auc_score(y_true, proba_positive))
    except Exception as exc:
        logger.warning("Could not compute ROC-AUC: %s", exc)
        return 0.0


def _roc_curve_data(
    y_true: np.ndarray, proba_positive: Optional[np.ndarray]
) -> Dict[str, List[float]]:
    """Return FPR/TPR/threshold samples suitable for plotting and JSON."""
    if proba_positive is None or len(np.unique(y_true)) < 2:
        return {"fpr": [], "tpr": [], "thresholds": []}
    try:
        fpr, tpr, thresholds = roc_curve(y_true, proba_positive)
    except Exception as exc:
        logger.warning("Could not compute ROC curve: %s", exc)
        return {"fpr": [], "tpr": [], "thresholds": []}
    return {
        "fpr": [float(value) for value in fpr],
        "tpr": [float(value) for value in tpr],
        "thresholds": [float(value) for value in thresholds],
    }


def _confusion_matrix_data(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    """Return a labelled 2x2 confusion matrix (genuine/fake, genuine/fake)."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "labels": [0, 1],
        "matrix": [[int(value) for value in row] for row in matrix],
    }


def _evaluate_one(model: Any, X_test: Any, y_true: np.ndarray) -> Dict[str, Any]:
    """Compute the full metric bundle for a single fitted model."""
    y_pred = np.asarray(model.predict(X_test))
    proba = _predict_proba(model, X_test)
    proba_positive = _positive_scores(proba)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, proba_positive),
        "roc_curve": _roc_curve_data(y_true, proba_positive),
        "confusion_matrix": _confusion_matrix_data(y_true, y_pred),
    }

    training_time = getattr(model, "training_time_seconds_", None)
    metrics["training_time"] = (
        float(training_time) if training_time is not None and not np.isnan(training_time) else None
    )

    logger.info(
        "%s -> acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f",
        model.__class__.__name__,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
    )
    return metrics


def evaluate_models(
    models: Dict[str, Any], X_test: Any, y_test: Sequence[Any]
) -> Dict[str, Any]:
    """Evaluate every classifier and return a full metrics bundle.

    Args:
        models: Dict mapping model names to fitted estimators.
        X_test: Feature matrix (dense or sparse) for the test split.
        y_test: Ground-truth labels for the test split.

    Returns:
        A dict keyed by model name, each entry containing ``accuracy``,
        ``precision``, ``recall``, ``f1``, ``roc_auc``, ``roc_curve``,
        ``confusion_matrix`` and ``training_time``, plus a ``metadata`` entry
        describing the evaluation run.
    """
    y_true = np.asarray(y_test)
    if X_test.shape[0] != y_true.shape[0]:
        raise ValueError(
            "X_test and y_test must have the same number of rows; "
            f"got {X_test.shape[0]} vs {y_true.shape[0]}"
        )

    results: Dict[str, Any] = {
        _METADATA_KEY: {
            "evaluated_at": datetime.utcnow().isoformat() + "Z",
            "test_samples": int(y_true.shape[0]),
            "feature_count": int(X_test.shape[1]),
            "model_names": list(models.keys()),
        }
    }
    for name, model in models.items():
        results[name] = _evaluate_one(model, X_test, y_true)
    return results


# --------------------------------------------------------------------------- #
# Database persistence
# --------------------------------------------------------------------------- #
def save_metrics_to_db(metrics: Dict[str, Any], db: Any = None) -> None:
    """Upsert evaluation metrics into the ``model_metrics`` table.

    Args:
        metrics: Output of :func:`evaluate_models`.
        db: Optional SQLAlchemy session; a fresh one is opened and closed when
            omitted.
    """
    owns_session = db is None
    if owns_session:
        db = SessionLocal()

    try:
        model_names = [name for name in metrics if name != _METADATA_KEY]
        for name in model_names:
            entry = metrics[name]
            row = (
                db.query(ModelMetric)
                .filter(ModelMetric.model_name == name)
                .first()
            )
            values: Dict[str, Any] = {
                "accuracy": entry.get("accuracy", 0.0),
                "precision_": entry.get("precision", 0.0),
                "recall": entry.get("recall", 0.0),
                "f1": entry.get("f1", 0.0),
                "roc_auc": entry.get("roc_auc", 0.0),
                "training_time": entry.get("training_time"),
            }
            if row is None:
                db.add(ModelMetric(model_name=name, **values))
            else:
                for key, value in values.items():
                    setattr(row, key, value)
            logger.info("Recorded metrics for model %r", name)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to persist model metrics")
        raise
    finally:
        if owns_session:
            db.close()


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _save_chart(fig: Any, path: Path) -> str:
    """Persist a figure with the dark theme and close it."""
    target = Path(path)
    try:
        fig.savefig(
            str(target),
            dpi=150,
            facecolor=PANEL_BLACK,
            bbox_inches="tight",
        )
        logger.info("Saved chart -> %s", target)
    finally:
        plt.close(fig)
    return str(target)


def _chart_metrics_comparison(metrics: Dict[str, Any], out_dir: Path) -> str:
    """Grouped bar chart of the headline metrics per model."""
    model_names = [name for name in metrics if name != _METADATA_KEY]
    positions = np.arange(len(_COMPARISON_METRICS))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for index, name in enumerate(model_names):
        values = [
            float(metrics[name].get(key, 0.0)) for key in _COMPARISON_METRICS
        ]
        offset = (index - (len(model_names) - 1) / 2) * bar_width
        bars = ax.bar(
            positions + offset,
            values,
            width=bar_width,
            label=name,
            color=_MODEL_PALETTE[index % len(_MODEL_PALETTE)],
            edgecolor=TEXT_LIGHT,
            linewidth=0.5,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=TEXT_LIGHT,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels([key.upper() for key in _COMPARISON_METRICS])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend(loc="lower right", ncols=len(model_names))
    ax.grid(axis="y")
    return _save_chart(fig, out_dir / "metrics_comparison.png")


def _chart_roc_curves(metrics: Dict[str, Any], out_dir: Path) -> str:
    """Overlay of the ROC curves with AUC annotations per model."""
    model_names = [name for name in metrics if name != _METADATA_KEY]
    fig, ax = plt.subplots(figsize=(8, 8))

    for index, name in enumerate(model_names):
        roc = metrics[name].get("roc_curve", {})
        fpr = roc.get("fpr", [])
        tpr = roc.get("tpr", [])
        auc = float(metrics[name].get("roc_auc", 0.0))
        if not fpr or not tpr:
            logger.warning("No ROC curve data for %r; skipping line", name)
            continue
        ax.plot(
            fpr,
            tpr,
            color=_MODEL_PALETTE[index % len(_MODEL_PALETTE)],
            linewidth=2,
            label=f"{name} (AUC = {auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color=TEXT_MUTED, linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    ax.grid()
    return _save_chart(fig, out_dir / "roc_curves.png")


def _chart_confusion_matrices(metrics: Dict[str, Any], out_dir: Path) -> str:
    """Confusion-matrix heatmap grid, one panel per model."""
    model_names = [name for name in metrics if name != _METADATA_KEY]
    columns = min(3, len(model_names)) if model_names else 1
    rows = int(np.ceil(len(model_names) / columns)) if model_names else 1

    fig, axes = plt.subplots(
        rows, columns, figsize=(5.2 * columns, 4.6 * rows), squeeze=False
    )
    flat_axes = axes.ravel()
    max_count = 1

    for index, name in enumerate(model_names):
        ax = flat_axes[index]
        cm = metrics[name].get("confusion_matrix", {})
        matrix = np.asarray(cm.get("matrix", [[0, 0], [0, 0]]), dtype=int)
        max_count = max(max_count, int(matrix.max()))

        ax.imshow(matrix, cmap="Greens", vmin=0, vmax=max_count)
        ax.set_title(name.replace("_", " ").title(), pad=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Genuine", "Fake"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Genuine", "Fake"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        half = max_count / 2
        for r in range(2):
            for c in range(2):
                ax.text(
                    c,
                    r,
                    str(matrix[r, c]),
                    ha="center",
                    va="center",
                    color="#000000" if matrix[r, c] >= half else TEXT_LIGHT,
                    fontsize=13,
                    weight="bold",
                )

    for ax in flat_axes[len(model_names):]:
        ax.set_visible(False)

    fig.suptitle("Confusion Matrices", y=0.98, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_chart(fig, out_dir / "confusion_matrices.png")


def generate_comparison_charts(
    metrics: Dict[str, Any], save_dir: Union[str, Path]
) -> List[str]:
    """Render the full set of comparison charts for the evaluated models.

    Charts use a black/green theme and are written to ``save_dir``:

    * ``metrics_comparison.png`` - grouped bar chart of headline metrics.
    * ``roc_curves.png``         - overlaid ROC curves with AUC labels.
    * ``confusion_matrices.png`` - heatmap grid of confusion matrices.

    Args:
        metrics: Output of :func:`evaluate_models`.
        save_dir: Directory the charts are written to (created if missing).

    Returns:
        Sorted list of full paths to the generated chart images.
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_theme()

    if not any(name != _METADATA_KEY for name in metrics):
        raise ValueError("metrics contains no model data; charts cannot be generated")

    generated = [
        _chart_metrics_comparison(metrics, out_dir),
        _chart_roc_curves(metrics, out_dir),
        _chart_confusion_matrices(metrics, out_dir),
    ]
    logger.info("Generated %d comparison chart(s) in %s", len(generated), out_dir)
    return sorted(generated)


# --------------------------------------------------------------------------- #
# End-to-end convenience
# --------------------------------------------------------------------------- #
def evaluate_pipeline(
    models: Dict[str, Any],
    X_test: Any,
    y_test: Sequence[Any],
    save_dir: Optional[Union[str, Path]] = None,
    persist_to_db: bool = True,
) -> Dict[str, Any]:
    """Evaluate, persist metrics and optionally render charts in one call.

    Args:
        models: Dict mapping model names to fitted estimators.
        X_test: Test feature matrix.
        y_test: Test ground-truth labels.
        save_dir: Optional chart directory; charts are generated when given.
        persist_to_db: When ``True`` metrics are recorded in the database.

    Returns:
        The full metrics bundle from :func:`evaluate_models`.
    """
    metrics = evaluate_models(models, X_test, y_test)

    if persist_to_db:
        save_metrics_to_db(metrics)

    if save_dir is not None:
        generate_comparison_charts(metrics, save_dir)

    return metrics