"""History, deletion and dashboard endpoints for the DocuGuard API.

Provides paginated access to past analyses with optional decision / text
filters, the ability to remove a single analysis (cascading to its marker and
model-prediction rows), and the aggregated statistics shown on the dashboard.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, selectinload

from app.api.analysis import _delete_explanation
from app.config import get_settings
from app.database.database import get_db
from app.models.database_models import Analysis, Document, Marker, ModelMetric
from app.models.schemas import (
    AnalysisDecision,
    DashboardStats,
    HistoryItem,
    HistoryResponse,
    MessageResponse,
    ModelPerformance,
    RiskDistributionBucket,
    SeverityDistribution,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

settings = get_settings()

_DECISION_VALUES = {"original", "suspicious", "forged"}

# Risk-band labels (mirrors the severity vocabulary of the dashboard buckets).
_LOW_BAND = "Low risk"
_MEDIUM_BAND = "Medium risk"
_HIGH_BAND = "High risk"


# --------------------------------------------------------------------------- #
# Serialisation helpers
# --------------------------------------------------------------------------- #
def _build_history_item(analysis: Analysis) -> HistoryItem:
    """Build a HistoryItem from an analysis row (markers preloaded)."""
    try:
        decision = AnalysisDecision(analysis.decision)
    except ValueError:
        decision = AnalysisDecision.SUSPICIOUS

    return HistoryItem(
        analysis_id=analysis.id,
        document_id=analysis.document_id,
        original_filename=analysis.document.original_filename,
        file_type=analysis.document.file_type,
        risk_score=analysis.risk_score,
        decision=decision,
        confidence=analysis.confidence,
        marker_count=len(analysis.markers),
        created_at=analysis.created_at,
    )


def _to_model_performance(metric: ModelMetric) -> ModelPerformance:
    """Build a ModelPerformance schema from a ModelMetric DB row."""
    return ModelPerformance(
        id=metric.id,
        model_name=metric.model_name,
        accuracy=metric.accuracy,
        precision=metric.precision_,
        recall=metric.recall,
        f1=metric.f1,
        roc_auc=metric.roc_auc,
        training_time=metric.training_time,
    )


def _analysis_query(
    decision: Optional[str],
    search: Optional[str],
) -> Any:
    """Build the base (joined) query for analyses with optional filters."""
    stmt = (
        select(Analysis)
        .join(Document, Analysis.document_id == Document.id)
        .options(selectinload(Analysis.document), selectinload(Analysis.markers))
    )
    if decision:
        stmt = stmt.where(Analysis.decision == decision)
    if search:
        pattern = f"%{search}%"
        stmt = stmt.where(
            or_(
                Document.original_filename.ilike(pattern),
                Document.file_type.ilike(pattern),
            )
        )
    return stmt


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.get("/history", response_model=HistoryResponse)
async def get_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    decision: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> HistoryResponse:
    """Return a paginated list of analyses with optional filters.

    ``decision`` accepts ``original``, ``suspicious`` or ``forged``.
    ``search`` performs a case-insensitive match against the original
    filename and the file type.
    """
    if decision is not None and decision.lower() not in _DECISION_VALUES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid decision '{decision}'. "
                "Allowed values: original, suspicious, forged"
            ),
        )
    decision = decision.lower() if decision else None
    search = search.strip() if search and search.strip() else None

    base = _analysis_query(decision, search)

    total_stmt = select(func.count()).select_from(base.subquery())
    total = db.execute(total_stmt).scalar_one() or 0

    rows = (
        db.execute(
            base.order_by(Analysis.created_at.desc()).offset(skip).limit(limit)
        )
        .scalars()
        .all()
    )

    items = [_build_history_item(analysis) for analysis in rows]
    logger.info("History query returned %d of %d analyses", len(items), total)
    return HistoryResponse(total=total, items=items)


@router.delete("/history/{analysis_id}", response_model=MessageResponse)
async def delete_analysis(
    analysis_id: int,
    db: Session = Depends(get_db),
) -> MessageResponse:
    """Delete an analysis and its related rows.

    The marker and model-prediction rows are removed via the relationship
    cascade, and any stored explainability payload is also deleted. The
    underlying document (and its file) is intentionally kept.
    """
    analysis = db.execute(
        select(Analysis).where(Analysis.id == analysis_id)
    ).scalar_one_or_none()
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    document_id = analysis.document_id
    db.delete(analysis)
    db.commit()
    _delete_explanation(analysis_id)
    logger.info(
        "Deleted analysis #%s (document #%s) and related data",
        analysis_id,
        document_id,
    )
    return MessageResponse(
        message="Analysis deleted successfully",
        detail=f"Analysis #{analysis_id} and its related data were removed",
    )


@router.get("/dashboard", response_model=DashboardStats)
async def get_dashboard_stats(db: Session = Depends(get_db)) -> DashboardStats:
    """Return aggregated statistics for the dashboard.

    Aggregations cover documents, analyses, markers, decisions, document
    types, marker severities, risk bands, recent analyses and registered
    model metrics.
    """
    total_documents = db.scalar(select(func.count(Document.id))) or 0
    total_analyses = db.scalar(select(func.count(Analysis.id))) or 0
    total_markers = db.scalar(select(func.count(Marker.id))) or 0
    average_risk_score = db.scalar(select(func.avg(Analysis.risk_score))) or 0.0

    decision_rows = db.execute(
        select(Analysis.decision, func.count()).group_by(Analysis.decision)
    ).all()
    decision_distribution: Dict[str, int] = {
        str(decision): int(count) for decision, count in decision_rows
    }

    doc_type_rows = db.execute(
        select(Document.document_type, func.count()).group_by(Document.document_type)
    ).all()
    document_type_distribution: Dict[str, int] = {
        str(doc_type): int(count) for doc_type, count in doc_type_rows
    }

    severity_rows = db.execute(
        select(Marker.severity, func.count()).group_by(Marker.severity)
    ).all()
    severity_counts: Dict[str, int] = {
        str(severity).lower(): int(count) for severity, count in severity_rows
    }
    severity_distribution = SeverityDistribution(
        low=severity_counts.get("low", 0),
        medium=severity_counts.get("medium", 0),
        high=severity_counts.get("high", 0),
        critical=severity_counts.get("critical", 0),
    )

    risk_scores = db.execute(select(Analysis.risk_score)).scalars().all()

    def _bucket(label: str, predicate: Any) -> RiskDistributionBucket:
        count = sum(1 for score in risk_scores if predicate(score))
        percentage = (
            round(count / total_analyses * 100.0, 1) if total_analyses else 0.0
        )
        return RiskDistributionBucket(label=label, count=count, percentage=percentage)

    risk_distribution = [
        _bucket(
            _LOW_BAND,
            lambda s: s < settings.medium_risk_threshold,
        ),
        _bucket(
            _MEDIUM_BAND,
            lambda s: (
                s >= settings.medium_risk_threshold
                and s < settings.high_risk_threshold
            ),
        ),
        _bucket(
            _HIGH_BAND,
            lambda s: s >= settings.high_risk_threshold,
        ),
    ]

    recent_rows = (
        db.execute(
            _analysis_query(None, None)
            .order_by(Analysis.created_at.desc())
            .limit(5)
        )
        .scalars()
        .all()
    )
    recent_analyses = [_build_history_item(analysis) for analysis in recent_rows]

    metric_rows = (
        db.execute(select(ModelMetric).order_by(ModelMetric.model_name)).scalars().all()
    )
    model_performance = [_to_model_performance(metric) for metric in metric_rows]

    week_start = datetime.utcnow() - timedelta(days=7)
    files_this_week = (
        db.scalar(
            select(func.count(Document.id)).where(Document.uploaded_at >= week_start)
        )
        or 0
    )
    analyses_this_week = (
        db.scalar(
            select(func.count(Analysis.id)).where(Analysis.created_at >= week_start)
        )
        or 0
    )

    logger.info(
        "Dashboard stats: %d documents, %d analyses, avg risk %.1f",
        total_documents,
        total_analyses,
        average_risk_score,
    )
    return DashboardStats(
        total_documents=total_documents,
        total_analyses=total_analyses,
        total_markers=total_markers,
        average_risk_score=round(float(average_risk_score), 2),
        decision_distribution=decision_distribution,
        document_type_distribution=document_type_distribution,
        severity_distribution=severity_distribution,
        risk_distribution=risk_distribution,
        recent_analyses=recent_analyses,
        model_performance=model_performance,
        files_this_week=files_this_week,
        analyses_this_week=analyses_this_week,
    )