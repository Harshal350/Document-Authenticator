"""Forensic analysis endpoints for the DocuGuard API.

An analysis runs the full detection pipeline against a previously uploaded
document: feature extraction, marker detection, machine-learning predictions,
risk scoring and explainability. Results are persisted to the ``analyses``,
``markers`` and ``model_predictions`` tables and returned as a complete
:class:`AnalysisResult`.

Marker category mapping
-----------------------
The marker detector emits *core* category strings (``textual``, ``numerical``,
``date``, ``id_reference``, ``structural``, ``metadata``, ``visual``). These
core strings are stored verbatim in the database. When a marker is exposed
through the :class:`MarkerResult` schema the core category is translated into
the richer :class:`MarkerCategory` enum (CONTENT / INCONSISTENCY / DIGITAL / …)
via :data:`MARKER_CATEGORY_MAP`.
"""

import inspect
import json
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.document_processor import DocumentProcessor
from app.core.document_type_detector import DocumentTypeDetector
from app.core.explainability import ExplainabilityEngine
from app.core.feature_engineering import FeatureEngineer
from app.core.marker_detector import MarkerDetector
from app.core.risk_engine import RiskEngine
from app.database.database import get_db
from app.ml.predict import ModelPredictor
from app.models.database_models import Analysis, Document, Marker, ModelPrediction
from app.models.schemas import (
    AnalysisDecision,
    AnalysisResult,
    DocumentInfo,
    DocumentType,
    MarkerCategory,
    MarkerResult,
    ModelPredictionResult,
    Severity,
    TopFactor,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter()

_document_processor = DocumentProcessor()
_feature_engineer = FeatureEngineer()
_marker_detector = MarkerDetector()
_model_predictor = ModelPredictor()
_risk_engine = RiskEngine()
_explainability_engine = ExplainabilityEngine()
_document_type_detector = DocumentTypeDetector()

# mtime signature of the loaded model artifacts; lets the API pick up a fresh
# training run without restarting the service.
_predictor_signature: float = 0.0

# --------------------------------------------------------------------------- #
# Mapping helpers: DB core categories  ->  schema enums
# --------------------------------------------------------------------------- #
MARKER_CATEGORY_MAP: Dict[str, MarkerCategory] = {
    "textual": MarkerCategory.CONTENT,
    "numerical": MarkerCategory.INCONSISTENCY,
    "date": MarkerCategory.INCONSISTENCY,
    "id_reference": MarkerCategory.DIGITAL,
    "structural": MarkerCategory.STRUCTURAL,
    "metadata": MarkerCategory.METADATA,
    "visual": MarkerCategory.VISUAL,
}

SEVERITY_MAP: Dict[str, Severity] = {
    "low": Severity.LOW,
    "medium": Severity.MEDIUM,
    "high": Severity.HIGH,
    "critical": Severity.CRITICAL,
}

DECISION_VALUES = {"original", "suspicious", "forged"}
_DECISION_ALIASES: Dict[str, str] = {
    "genuine": "original",
    "authentic": "original",
    "legitimate": "original",
    "fake": "forged",
    "fraudulent": "forged",
    "forgery": "forged",
}


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def _field(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict or an attribute-bearing object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _invoke(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Call ``func`` passing only the keyword arguments it declares.

    Used to tolerate slightly different signatures on the ML / explainability
    modules. If the accepted-argument call fails (signature mismatch) the
    function is retried once without the unsupported keyword arguments.
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


def _clamp(value: Any, low: float, high: float, default: float) -> float:
    """Coerce a value to a float bounded by ``[low, high]``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _normalize_decision(value: Any, risk_score: Optional[float] = None) -> str:
    """Normalise a decision string into one of the DB/accepted values."""
    raw = str(value or "").strip().lower()
    if raw in DECISION_VALUES:
        return raw
    raw = _DECISION_ALIASES.get(raw, raw)
    if raw in DECISION_VALUES:
        return raw

    score = _clamp(risk_score, 0.0, 100.0, 0.0) if risk_score is not None else 0.0
    if score >= settings.high_risk_threshold:
        return "forged"
    if score >= settings.medium_risk_threshold:
        return "suspicious"
    return "original"


def _map_marker_category(category: Any) -> MarkerCategory:
    """Map a DB-stored core category string onto the schema enum."""
    if not category:
        return MarkerCategory.OTHER
    return MARKER_CATEGORY_MAP.get(str(category).strip().lower(), MarkerCategory.OTHER)


def _map_severity(severity: Any) -> Severity:
    """Map a stored severity string onto the schema enum."""
    if not severity:
        return Severity.MEDIUM
    return SEVERITY_MAP.get(str(severity).strip().lower(), Severity.MEDIUM)


def _normalize_prediction(raw: Any) -> Dict[str, Any]:
    """Normalise a raw prediction dict so probabilities are valid and sum to ~1."""
    genuine = _clamp(
        _field(raw, "genuine_probability", _field(raw, "genuine_prob")),
        0.0,
        1.0,
        0.0,
    )
    fake = _clamp(
        _field(raw, "fake_probability", _field(raw, "fake_prob")),
        0.0,
        1.0,
        0.0,
    )
    if genuine == 0.0 and fake == 0.0:
        genuine, fake = 0.5, 0.5
    if abs(genuine + fake - 1.0) > 0.02:
        fake = round(max(0.0, min(1.0, 1.0 - genuine)), 6)
        genuine = round(max(0.0, min(1.0, genuine)), 6)
    return {
        "model_name": str(_field(raw, "model_name", "unknown")),
        "genuine_probability": round(genuine, 6),
        "fake_probability": round(fake, 6),
    }


def _run_predictions(text: str, feature_vector: Any) -> List[Dict[str, Any]]:
    """Run the ML models against the document, degrading gracefully.

    ``ModelPredictor.predict`` returns a mapping of ``model_name ->
    {genuine_prob, fake_prob}``.  When a TF-IDF vectorizer was persisted with
    the models the combined TF-IDF + numeric path is used so the input width
    matches the training matrix.
    """
    if not settings.enable_ml:
        logger.warning("ML is disabled; skipping model predictions")
        return []
    try:
        global _predictor_signature
        signature = _model_predictor.artifacts_signature()
        if signature != _predictor_signature:
            _model_predictor.reload()
            _predictor_signature = signature

        predictor = _model_predictor
        if not predictor.is_available:
            logger.info("No trained models available yet; skipping predictions")
            return []

        raw: Any = None
        if predictor.vectorizer is not None and len(feature_vector):
            try:
                raw = predictor.predict_from_text_and_features(text, feature_vector)
            except Exception as exc:
                logger.warning(
                    "Combined text+feature prediction failed (%s); "
                    "falling back to feature vector",
                    exc,
                )
        if raw is None:
            raw = predictor.predict(feature_vector)
        if raw is None:
            return []

        if isinstance(raw, dict):
            items = [
                {"model_name": name, **(value if isinstance(value, dict) else {})}
                for name, value in raw.items()
            ]
        elif isinstance(raw, list):
            items = raw
        else:
            return []

        predictions = [
            _normalize_prediction(item)
            for item in items
            if isinstance(item, dict)
        ]
        logger.info("Produced %d model predictions", len(predictions))
        return predictions
    except Exception as exc:
        logger.warning("Model prediction failed, continuing without ML: %s", exc)
        return []


# --------------------------------------------------------------------------- #
# Risk helpers
# --------------------------------------------------------------------------- #
def _get_risk_fields(risk_result: Any) -> Dict[str, Any]:
    """Extract analysis columns from the risk-engine result, defensively."""
    risk_score = _clamp(_field(risk_result, "risk_score"), 0.0, 100.0, 0.0)
    decision = _normalize_decision(_field(risk_result, "decision"), risk_score)
    return {
        "risk_score": risk_score,
        "decision": decision,
        "confidence": _clamp(_field(risk_result, "confidence"), 0.0, 1.0, 0.0),
        "ml_risk_score": _clamp(
            _field(risk_result, "ml_risk_score"), 0.0, 100.0, 0.0
        ),
        "marker_risk_score": _clamp(
            _field(risk_result, "marker_risk_score"), 0.0, 100.0, 0.0
        ),
        "structural_risk_score": _clamp(
            _field(risk_result, "structural_risk_score"), 0.0, 100.0, 0.0
        ),
        "consistency_risk_score": _clamp(
            _field(risk_result, "consistency_risk_score"), 0.0, 100.0, 0.0
        ),
    }


def _fallback_risk(
    predictions: List[Dict[str, Any]],
    markers: List[Dict[str, Any]],
    stats: Dict[str, Any],
    features: Dict[str, Any],
) -> Dict[str, Any]:
    """Heuristic risk estimate used when the risk engine is unavailable."""
    marker_scores = [_clamp(m.get("score"), 0.0, 1.0, 0.0) for m in markers]
    marker_risk = (
        float(np.mean(marker_scores)) * 100.0 if marker_scores else 0.0
    )

    structural_scores = [
        _clamp(m.get("score"), 0.0, 1.0, 0.0)
        for m in markers
        if m.get("category") == "structural"
    ]
    structural_risk = (
        float(np.mean(structural_scores)) * 100.0 if structural_scores else 0.0
    )

    consistency_scores = [
        _clamp(m.get("score"), 0.0, 1.0, 0.0)
        for m in markers
        if m.get("category") in {"textual", "numerical", "date", "id_reference"}
    ]
    consistency_risk = (
        float(np.mean(consistency_scores)) * 100.0
        if consistency_scores
        else 0.0
    )

    fake_probs = [
        _clamp(p.get("fake_probability"), 0.0, 1.0, 0.0) for p in predictions
    ]
    ml_risk = float(np.mean(fake_probs)) * 100.0 if fake_probs else 0.0

    risk_score = round(
        min(100.0, 0.5 * ml_risk + 0.35 * marker_risk + 0.15 * structural_risk), 2
    )
    decision = _normalize_decision(None, risk_score)
    confidence = round(
        max(0.5, min(1.0, 1.0 - abs(risk_score - 50.0) / 50.0)), 2
    )
    return {
        "risk_score": risk_score,
        "decision": decision,
        "confidence": confidence,
        "ml_risk_score": round(ml_risk, 2),
        "marker_risk_score": round(marker_risk, 2),
        "structural_risk_score": round(structural_risk, 2),
        "consistency_risk_score": round(consistency_risk, 2),
    }


# --------------------------------------------------------------------------- #
# Serialisation helpers
# --------------------------------------------------------------------------- #
def _update_consistency_features(
    features: Dict[str, Any],
    stats: Dict[str, Any],
    markers: List[Dict[str, Any]],
) -> None:
    """Refresh the consistency features from the detected marker set.

    Date / numerical markers feed ``total_inconsistencies`` and
    ``id_reference`` markers feed ``id_inconsistencies``.  Both the feature
    dictionary and the statistics dictionary are updated so every consumer
    (feature vector, risk engine, marker engine) sees the same values.
    """
    date_categories = {"date", "dates"}
    number_categories = {"numerical", "number", "numbers"}
    id_categories = {"id_reference", "id", "reference"}

    date_count = 0
    number_count = 0
    id_count = 0
    for marker in markers:
        category = str(marker.get("category", "")).strip().lower()
        if category in date_categories:
            date_count += 1
        elif category in number_categories:
            number_count += 1
        elif category in id_categories:
            id_count += 1

    total = date_count + number_count
    features["total_inconsistencies"] = float(total)
    features["id_inconsistencies"] = float(id_count)
    stats["total_inconsistencies"] = total
    stats["id_inconsistencies"] = id_count


def _to_marker_result(marker: Marker) -> MarkerResult:
    """Build a schema MarkerResult from a DB marker row (category mapped)."""
    return MarkerResult(
        id=marker.id,
        category=_map_marker_category(marker.category),
        name=marker.name,
        severity=_map_severity(marker.severity),
        description=marker.description or "",
        page_number=marker.page_number,
        evidence=marker.evidence or "",
        score=_clamp(marker.score, 0.0, 100.0, 0.0),
    )


def _to_prediction_result(prediction: ModelPrediction) -> ModelPredictionResult:
    """Build a schema ModelPredictionResult from a DB prediction row."""
    raw = {
        "model_name": prediction.model_name,
        "genuine_probability": prediction.genuine_probability,
        "fake_probability": prediction.fake_probability,
    }
    normalized = _normalize_prediction(raw)
    return ModelPredictionResult(
        id=prediction.id,
        model_name=normalized["model_name"],
        genuine_probability=normalized["genuine_probability"],
        fake_probability=normalized["fake_probability"],
    )


def _to_document_info(document: Optional[Document]) -> Optional[DocumentInfo]:
    """Build a schema DocumentInfo from a DB document row."""
    if document is None:
        return None
    try:
        document_type = DocumentType(document.document_type)
    except ValueError:
        document_type = DocumentType.UNKNOWN
    return DocumentInfo(
        id=document.id,
        original_filename=document.original_filename,
        file_type=document.file_type,
        file_size=document.file_size,
        pages=document.pages,
        words=document.words,
        document_type=document_type,
        uploaded_at=document.uploaded_at,
    )


def _to_analysis_result(
    analysis: Analysis,
    explanation: Optional[Dict[str, Any]] = None,
) -> AnalysisResult:
    """Build a complete AnalysisResult from the persisted analysis."""
    try:
        decision = AnalysisDecision(analysis.decision)
    except ValueError:
        decision = AnalysisDecision.SUSPICIOUS

    if explanation is None:
        explanation = _load_explanation(analysis.id) or {}
    summary = explanation.get("summary") if isinstance(explanation, dict) else None
    raw_factors = (
        explanation.get("top_factors", [])
        if isinstance(explanation, dict)
        else []
    )
    top_factors: List[TopFactor] = []
    for factor in raw_factors if isinstance(raw_factors, list) else []:
        if not isinstance(factor, dict):
            continue
        top_factors.append(
            TopFactor(
                factor=str(factor.get("factor", factor.get("label", "factor"))),
                contribution=float(factor.get("contribution", 0.0) or 0.0),
                direction=str(factor.get("direction", "neutral")),
                detail=str(factor.get("detail", "") or ""),
            )
        )

    return AnalysisResult(
        id=analysis.id,
        document_id=analysis.document_id,
        risk_score=_clamp(analysis.risk_score, 0.0, 100.0, 0.0),
        decision=decision,
        confidence=_clamp(analysis.confidence, 0.0, 1.0, 0.0),
        ml_risk_score=_clamp(analysis.ml_risk_score, 0.0, 100.0, 0.0),
        marker_risk_score=_clamp(analysis.marker_risk_score, 0.0, 100.0, 0.0),
        structural_risk_score=_clamp(analysis.structural_risk_score, 0.0, 100.0, 0.0),
        consistency_risk_score=_clamp(analysis.consistency_risk_score, 0.0, 100.0, 0.0),
        created_at=analysis.created_at,
        markers=[_to_marker_result(marker) for marker in analysis.markers],
        model_predictions=[
            _to_prediction_result(prediction)
            for prediction in analysis.model_predictions
        ],
        document=_to_document_info(analysis.document),
        explanation=summary,
        top_factors=top_factors,
    )


# --------------------------------------------------------------------------- #
# Explainability persistence
# --------------------------------------------------------------------------- #
def _explainability_dir() -> Any:
    """Return (creating if needed) the directory for explanation payloads."""
    directory = settings.report_dir_path / "explainability"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _save_explanation(analysis_id: int, explanation: Any) -> None:
    """Persist the engine explanation as JSON next to the reports directory."""
    if explanation is None:
        return
    try:
        if isinstance(explanation, str):
            payload: Dict[str, Any] = {"summary": explanation}
        elif hasattr(explanation, "model_dump"):
            payload = explanation.model_dump()
        elif hasattr(explanation, "dict"):
            payload = explanation.dict()
        elif not isinstance(explanation, dict):
            payload = {"raw": str(explanation)}
        else:
            payload = explanation

        target = _explainability_dir() / f"{analysis_id}.json"
        target.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("Saved explainability payload for analysis #%s", analysis_id)
    except Exception as exc:
        logger.warning("Could not persist explanation: %s", exc)


def _load_explanation(analysis_id: int) -> Optional[Dict[str, Any]]:
    """Load a previously-persisted explanation payload (or ``None``)."""
    target = _explainability_dir() / f"{analysis_id}.json"
    if not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"summary": str(data)}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read explanation for analysis #%s: %s", analysis_id, exc)
        return None


def _delete_explanation(analysis_id: int) -> None:
    """Remove the stored explanation payload for an analysis."""
    target = _explainability_dir() / f"{analysis_id}.json"
    try:
        target.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not delete explanation for analysis #%s: %s", analysis_id, exc)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.post("/analyze/{document_id}", response_model=AnalysisResult)
async def analyze_document(
    document_id: int,
    db: Session = Depends(get_db),
) -> AnalysisResult:
    """Run the complete forensic pipeline on a previously uploaded document."""
    document = db.execute(
        select(Document).where(Document.id == document_id)
    ).scalar_one_or_none()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = settings.temp_dir_path / document.filename
    if not file_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Stored file for document #{document_id} is missing",
        )

    # 1. (Re)process the document to obtain fresh text / metadata / stats.
    processed = _document_processor.process_document(str(file_path), document.file_type)
    if not processed.get("success"):
        raise HTTPException(
            status_code=400,
            detail=processed.get("error") or "Document could not be processed",
        )

    text = processed.get("text") or ""
    metadata = processed.get("metadata") or {}
    stats = processed.get("stats") or {}

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text was detected. The document may require OCR "
                "or may contain protected content."
            ),
        )

    # 2. Feature engineering.
    features = _feature_engineer.extract_features(text, metadata, stats)
    try:
        feature_vector = _feature_engineer.create_feature_vector(text, metadata, stats)
    except Exception as exc:
        logger.warning("Could not build feature vector, continuing without ML: %s", exc)
        feature_vector = np.array([], dtype=np.float64)

    # 3. Marker detection.
    markers = _marker_detector.detect_all_markers(
        text, metadata, stats, feature_vector
    )
    markers = [m for m in markers if isinstance(m, dict)]
    logger.info("Detected %d markers for document #%s", len(markers), document_id)

    # 3b. Fold marker-derived inconsistency counts back into the feature set so
    #     the consistency features reflect what was actually detected.
    try:
        _update_consistency_features(features, stats, markers)
        if len(feature_vector):
            feature_vector = _feature_engineer.create_feature_vector(
                text, metadata, stats
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not refresh consistency features: %s", exc)

    # 3c. Semantic document type detection (invoice, certificate, ...).
    try:
        detected_type = _document_type_detector.detect(
            text, document.original_filename
        )
        if detected_type and detected_type != document.document_type:
            document.document_type = detected_type
            db.add(document)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Document type detection failed: %s", exc)

    # 4. Machine-learning predictions (degraded gracefully if ML is off).
    predictions = _run_predictions(text, feature_vector)

    # 5. Risk scoring.
    try:
        risk_result = _risk_engine.calculate_risk(predictions, markers, stats, features)
    except Exception as exc:
        logger.warning("Risk engine failed, using fallback estimate: %s", exc)
        risk_result = _fallback_risk(predictions, markers, stats, features)
    risk_fields = _get_risk_fields(risk_result)

    # 6. Persist the analysis, markers and model predictions.
    analysis = Analysis(document_id=document_id, **risk_fields)
    db.add(analysis)
    db.flush()

    for marker in markers:
        score = _clamp(marker.get("score"), 0.0, 1.0, 0.0) * 100.0
        db.add(
            Marker(
                analysis_id=analysis.id,
                category=str(marker.get("category", "other")).strip().lower(),
                name=str(marker.get("name", "unknown_marker"))[:120],
                severity=str(marker.get("severity", "medium")).strip().lower(),
                description=str(marker.get("description", "") or ""),
                page_number=(
                    int(marker["page_number"])
                    if marker.get("page_number") is not None
                    else None
                ),
                evidence=str(marker.get("evidence", "") or ""),
                score=round(score, 2),
            )
        )

    for prediction in predictions:
        db.add(
            ModelPrediction(
                analysis_id=analysis.id,
                model_name=prediction["model_name"],
                genuine_probability=prediction["genuine_probability"],
                fake_probability=prediction["fake_probability"],
            )
        )

    # 7. Explainability, persisted alongside the report directory.
    try:
        explanation = _explainability_engine.explain(
            risk_score=risk_fields["risk_score"],
            decision=risk_fields["decision"],
            model_predictions=predictions,
            markers=markers,
            feature_dict=features,
            stats=stats,
        )
        if explanation is None:
            explanation = {
                "summary": (
                    f"Combined risk score {risk_fields['risk_score']:.1f} "
                    f"({risk_fields['decision']}) based on {len(markers)} markers "
                    f"and {len(predictions)} model predictions."
                )
            }
    except Exception as exc:
        logger.warning("Explainability engine failed: %s", exc)
        explanation = {
            "summary": (
                f"Combined risk score {risk_fields['risk_score']:.1f} "
                f"({risk_fields['decision']}) based on {len(markers)} markers "
                f"and {len(predictions)} model predictions."
            )
        }
    _save_explanation(analysis.id, explanation)

    db.commit()
    db.refresh(analysis)
    logger.info(
        "Analysis #%s created for document #%s (decision=%s, risk=%.1f)",
        analysis.id,
        document_id,
        analysis.decision,
        analysis.risk_score,
    )
    return _to_analysis_result(analysis, explanation)


@router.get("/results/{analysis_id}", response_model=AnalysisResult)
async def get_results(
    analysis_id: int,
    db: Session = Depends(get_db),
) -> AnalysisResult:
    """Return the complete persisted result of a previous analysis."""
    analysis = db.execute(
        select(Analysis).where(Analysis.id == analysis_id)
    ).scalar_one_or_none()
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return _to_analysis_result(analysis)