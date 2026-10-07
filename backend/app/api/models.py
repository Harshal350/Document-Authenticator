"""Model management endpoints for the DocuGuard API.

Exposes model performance metrics persisted in the ``model_metrics`` table and
a training endpoint that loads the labelled dataset, trains the three models
(random forest, logistic regression and XGBoost), evaluates them, generates
comparison charts and records the resulting metrics back into the database.
"""

import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import PROJECT_ROOT, get_settings
from app.database.database import get_db
from app.ml.evaluate import evaluate_models, generate_comparison_charts
from app.ml.train import train_all_models, train_from_dataset
from app.models.database_models import ModelMetric
from app.models.schemas import ModelPerformance, ModelPerformanceList
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter()

_DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "training" / "documents.csv"


def _invoke(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Call ``func`` passing only the keyword arguments it declares.

    Tolerates small signature differences in the ML modules. When a keyword
    argument is not accepted it is dropped; if the corrected call still fails
    the function is retried once without any keyword arguments.
    """
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args, **kwargs)

    params = signature.parameters
    accepts_kwargs = any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()
    )
    bounded = (
        kwargs if accepts_kwargs else {k: v for k, v in kwargs.items() if k in params}
    )
    try:
        return func(*args, **bounded)
    except TypeError:
        if bounded or args:
            return func()
        raise


def _value(obj: Any, key: str, default: Any = None) -> Any:
    """Read a named field from a dict or an attribute-bearing object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_float(value: Any, default: float = 0.0) -> float:
    """Coerce a value to a float using *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_dataset_path(dataset_path: Optional[str]) -> Path:
    """Resolve the (optional) dataset path against the project default."""
    if dataset_path:
        candidate = Path(dataset_path).expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file not found: {dataset_path}",
        )
    if not _DEFAULT_DATASET_PATH.is_file():
        # mirror the repository layout ("data/training/documents.csv")
        raise HTTPException(
            status_code=404,
            detail=(
                f"Default dataset not found at {_DEFAULT_DATASET_PATH}. "
                "Pass an explicit dataset_path or place the labelled data at "
                "data/training/documents.csv."
            ),
        )
    return _DEFAULT_DATASET_PATH


def _persist_metrics(db: Session, metrics: Any) -> int:
    """Upsert evaluation metrics into the ``model_metrics`` table.

    Returns the number of models whose metrics were persisted.
    """
    if metrics is None:
        return 0

    if isinstance(metrics, dict):
        entries = []
        for key, val in metrics.items():
            if key == "metadata" or not isinstance(val, dict):
                continue
            entry = dict(val)
            entry["model_name"] = key
            entries.append(entry)
    elif isinstance(metrics, (list, tuple)):
        entries = list(metrics)
    else:
        entries = [metrics]

    saved = 0
    for entry in entries:
        if not isinstance(entry, (dict,)):
            if entry is None:
                continue
            model_name = _value(entry, "model_name") or _value(entry, "name")
            accuracy = _as_float(_value(entry, "accuracy"))
            precision = _as_float(_value(entry, "precision", _value(entry, "precision_")))
            recall = _as_float(_value(entry, "recall"))
            f1 = _as_float(_value(entry, "f1"))
            roc_auc = _as_float(_value(entry, "roc_auc", _value(entry, "auc")))
            training_time = _value(entry, "training_time")
        else:
            model_name = entry.get("model_name") or entry.get("name")
            accuracy = _as_float(entry.get("accuracy"))
            precision = _as_float(entry.get("precision", entry.get("precision_")))
            recall = _as_float(entry.get("recall"))
            f1 = _as_float(entry.get("f1"))
            roc_auc = _as_float(entry.get("roc_auc", entry.get("auc")))
            training_time = entry.get("training_time")

        if not model_name:
            logger.warning("Skipping metric entry without a model name: %r", entry)
            continue

        existing = db.execute(
            select(ModelMetric).where(ModelMetric.model_name == str(model_name))
        ).scalar_one_or_none()

        training_time_value = (
            _as_float(training_time) if training_time is not None else None
        )
        if existing is not None:
            existing.accuracy = accuracy
            existing.precision_ = precision
            existing.recall = recall
            existing.f1 = f1
            existing.roc_auc = roc_auc
            existing.training_time = training_time_value
        else:
            db.add(
                ModelMetric(
                    model_name=str(model_name),
                    accuracy=accuracy,
                    precision_=precision,
                    recall=recall,
                    f1=f1,
                    roc_auc=roc_auc,
                    training_time=training_time_value,
                )
            )
        saved += 1
        logger.info("Recorded metrics for model '%s'", model_name)

    return saved


@router.get("/models/performance", response_model=ModelPerformanceList)
async def get_model_performance(
    db: Session = Depends(get_db),
) -> ModelPerformanceList:
    """Return the persisted evaluation metrics for every trained model."""
    metric_rows = (
        db.execute(select(ModelMetric).order_by(ModelMetric.model_name)).scalars().all()
    )
    models = [
        ModelPerformance(
            id=metric.id,
            model_name=metric.model_name,
            accuracy=metric.accuracy,
            precision=metric.precision_,
            recall=metric.recall,
            f1=metric.f1,
            roc_auc=metric.roc_auc,
            training_time=metric.training_time,
        )
        for metric in metric_rows
    ]
    logger.info("Returned performance data for %d models", len(models))
    return ModelPerformanceList(total_models=len(models), models=models)


@router.post("/models/train")
async def train_models_endpoint(
    dataset_path: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Train, evaluate and persist the three detection models.

    The labelled dataset is loaded from *dataset_path* (defaults to
    ``data/training/documents.csv``), the models are retrained, evaluated and
    comparison charts are generated. Evaluation metrics are upserted into the
    ``model_metrics`` table before the training summary is returned.
    """
    dataset = _resolve_dataset_path(dataset_path)

    try:
        metrics, summary = train_from_dataset(dataset_path=str(dataset))
    except Exception as exc:
        logger.exception("Training failed for %s", dataset)
        raise HTTPException(
            status_code=500, detail=f"Model training failed: {exc}"
        )

    saved = _persist_metrics(db, metrics)
    db.commit()

    logger.info(
        "Training completed for %s: %d metric sets saved",
        dataset,
        saved,
    )
    return {
        "message": "Model training completed successfully",
        "dataset_path": str(dataset),
        "training_results": summary,
        "metrics_saved": saved,
        "charts_generated": bool(summary.get("charts_generated")),
    }