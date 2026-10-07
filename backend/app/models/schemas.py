"""Pydantic schemas used for request / response serialisation.

All schemas derive from :class:`DocuGuardSchema` which enables automatic
*ORM-compatible* mode (``from_attributes``) so that Pydantic can build model
instances directly from SQLAlchemy model objects.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DocuGuardSchema(BaseModel):
    """Shared configuration for every schema in the application."""

    model_config = ConfigDict(from_attributes=True, use_enum_values=True, populate_by_name=True)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DocumentType(str, Enum):
    """Semantic category detected for an uploaded document (spec section 6)."""

    INVOICE = "invoice"
    RECEIPT = "receipt"
    CERTIFICATE = "certificate"
    IDENTITY = "identity"
    EDUCATIONAL = "educational"
    EMPLOYMENT = "employment"
    FINANCIAL = "financial"
    APPLICATION = "application"
    AGREEMENT = "agreement"
    LETTER = "letter"
    GOVERNMENT = "government"
    UNKNOWN = "unknown"
    # Legacy values that may exist in older rows.
    ORIGINAL = "original"
    FORGED = "forged"
    SUSPICIOUS = "suspicious"


class AnalysisDecision(str, Enum):
    ORIGINAL = "original"
    FORGED = "forged"
    SUSPICIOUS = "suspicious"


class MarkerCategory(str, Enum):
    METADATA = "metadata"
    VISUAL = "visual"
    STRUCTURAL = "structural"
    CONTENT = "content"
    DIGITAL = "digital"
    INCONSISTENCY = "inconsistency"
    FONT = "font"
    LAYOUT = "layout"
    PERMISSION = "permission"
    OTHER = "other"


class ReportFormat(str, Enum):
    PDF = "pdf"
    HTML = "html"
    JSON = "json"


class SeverityDistribution(DocuGuardSchema):
    low: int = 0
    medium: int = 0
    high: int = 0
    critical: int = 0


# --------------------------------------------------------------------------- #
# Document schemas
# --------------------------------------------------------------------------- #
class DocumentUpload(DocuGuardSchema):
    """Full representation of a document row."""

    id: int
    filename: str
    original_filename: str
    file_type: str
    file_size: int
    pages: int = 0
    words: int = 0
    characters: int = 0
    tables_detected: int = 0
    ocr_used: bool = False
    document_type: DocumentType = DocumentType.UNKNOWN
    uploaded_at: datetime


class DocumentInfo(DocuGuardSchema):
    """Lightweight representation used in listings and nested payloads."""

    id: int
    original_filename: str
    file_type: str
    file_size: int
    pages: int = 0
    words: int = 0
    document_type: DocumentType = DocumentType.UNKNOWN
    uploaded_at: datetime


class DocumentUploadRequest(DocuGuardSchema):
    """Client-supplied metadata alongside a file upload."""

    original_filename: str = Field(..., min_length=1, max_length=255)
    file_type: str = Field(..., min_length=1, max_length=50)
    file_size: int = Field(..., gt=0)
    document_type: DocumentType = DocumentType.UNKNOWN


class UploadResponse(DocuGuardSchema):
    message: str = "File uploaded successfully"
    document_id: int
    filename: str


class BatchUploadResponse(DocuGuardSchema):
    message: str
    uploaded_count: int
    failed_count: int
    failed_files: List[Dict[str, str]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Marker schemas
# --------------------------------------------------------------------------- #
class MarkerResult(DocuGuardSchema):
    """A single forensic marker / finding."""

    id: Optional[int] = None
    category: MarkerCategory
    name: str
    severity: Severity
    description: str = ""
    page_number: Optional[int] = None
    evidence: str = ""
    score: float = Field(0.0, ge=0.0, le=100.0)

    @field_validator("score")
    @classmethod
    def _clamp_score(cls, value: float) -> float:
        return max(0.0, min(100.0, value))


class MarkerCreate(DocuGuardSchema):
    """Input payload for creating a marker."""

    category: MarkerCategory
    name: str = Field(..., min_length=1, max_length=120)
    severity: Severity = Severity.MEDIUM
    description: str = ""
    page_number: Optional[int] = None
    evidence: str = ""
    score: float = Field(0.0, ge=0.0, le=100.0)


# --------------------------------------------------------------------------- #
# Model prediction schemas
# --------------------------------------------------------------------------- #
class ModelPredictionResult(DocuGuardSchema):
    """Prediction contributed by a single ML model."""

    id: Optional[int] = None
    model_name: str
    genuine_probability: float = Field(..., ge=0.0, le=1.0)
    fake_probability: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _probabilities_sum_check(self):
        total = self.genuine_probability + self.fake_probability
        if abs(total - 1.0) > 0.02:
            raise ValueError(
                f"genuine_probability ({self.genuine_probability}) + "
                f"fake_probability ({self.fake_probability}) should sum to ~1.0, "
                f"got {total:.4f}"
            )
        return self


class ModelPredictionCreate(DocuGuardSchema):
    """Input payload for recording a model prediction."""

    model_name: str = Field(..., min_length=1, max_length=50)
    genuine_probability: float = Field(..., ge=0.0, le=1.0)
    fake_probability: float = Field(..., ge=0.0, le=1.0)


# --------------------------------------------------------------------------- #
# Analysis schemas
# --------------------------------------------------------------------------- #
class TopFactor(DocuGuardSchema):
    """A single explainability factor contributing to the risk score."""

    factor: str
    contribution: float = 0.0
    direction: str = "neutral"
    detail: str = ""


class AnalysisResult(DocuGuardSchema):
    """Complete forensic analysis result."""

    id: int
    document_id: int
    risk_score: float = Field(0.0, ge=0.0, le=100.0)
    decision: AnalysisDecision
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    ml_risk_score: float = Field(0.0, ge=0.0, le=100.0)
    marker_risk_score: float = Field(0.0, ge=0.0, le=100.0)
    structural_risk_score: float = Field(0.0, ge=0.0, le=100.0)
    consistency_risk_score: float = Field(0.0, ge=0.0, le=100.0)
    created_at: datetime

    markers: List[MarkerResult] = Field(default_factory=list)
    model_predictions: List[ModelPredictionResult] = Field(default_factory=list)
    document: Optional[DocumentInfo] = None
    explanation: Optional[str] = None
    top_factors: List[TopFactor] = Field(default_factory=list)


class AnalysisCreate(DocuGuardSchema):
    """Payload accepted when triggering an analysis."""

    document_id: int = Field(..., gt=0)
    models: List[str] = Field(default_factory=lambda: ["random_forest", "xgboost"])


class AnalysisBrief(DocuGuardSchema):
    """Short summary of an analysis used in history listings."""

    analysis_id: int
    document_id: int
    original_filename: str
    file_type: str
    risk_score: float
    decision: AnalysisDecision
    confidence: float
    marker_count: int
    created_at: datetime


# --------------------------------------------------------------------------- #
# Model performance / metrics schemas
# --------------------------------------------------------------------------- #
class ModelPerformanceBase(DocuGuardSchema):
    """Base fields shared by all model-performance representations."""

    model_name: str
    accuracy: float = Field(0.0, ge=0.0, le=1.0)
    precision: float = Field(0.0, ge=0.0, le=1.0)
    recall: float = Field(0.0, ge=0.0, le=1.0)
    f1: float = Field(0.0, ge=0.0, le=1.0)
    roc_auc: float = Field(0.0, ge=0.0, le=1.0)
    training_time: Optional[float] = None

    @field_validator("precision")
    @classmethod
    def _clamp(cls, value: float) -> float:
        return max(0.0, min(1.0, value))


class ModelPerformance(DocuGuardSchema):
    """Evaluation metrics for a trained / registered model."""

    id: Optional[int] = None
    model_name: str
    accuracy: float = 0.0
    precision: float = Field(
        0.0,
        validation_alias="precision_",
        serialization_alias="precision",
        ge=0.0,
        le=1.0,
    )
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    training_time: Optional[float] = None
    # Rich evaluation details (present after a training run) used by the
    # model-performance dashboard for ROC curves and confusion matrices.
    roc_curve: Optional[Dict[str, List[float]]] = None
    confusion_matrix: Optional[Dict[str, Any]] = None


class TrainingDatasetInfo(DocuGuardSchema):
    """Summary of the labelled dataset used for a training run."""

    path: str = ""
    rows: int = 0
    text_column: str = "text"
    label_column: str = "label"
    class_distribution: Dict[str, int] = Field(default_factory=dict)
    training_samples: int = 0
    testing_samples: int = 0
    feature_count: int = 0
    is_demo_data: bool = False


class TrainingResponse(DocuGuardSchema):
    """Response returned by the model training endpoint."""

    message: str = "Model training completed successfully"
    dataset_path: str = ""
    dataset: TrainingDatasetInfo = Field(default_factory=TrainingDatasetInfo)
    metrics_saved: int = 0
    charts_generated: bool = False
    model_names: List[str] = Field(default_factory=list)
    total_training_time_seconds: float = 0.0


class ModelPerformanceCreate(ModelPerformanceBase):
    """Payload accepted when recording new model metrics."""

    pass


class ModelPerformanceList(DocuGuardSchema):
    total_models: int
    models: List[ModelPerformance] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #
class HistoryItem(DocuGuardSchema):
    """Single entry in the analysis history feed."""

    analysis_id: int
    document_id: int
    original_filename: str
    file_type: str
    risk_score: float
    decision: AnalysisDecision
    confidence: float
    marker_count: int
    created_at: datetime


class HistoryResponse(DocuGuardSchema):
    total: int
    items: List[HistoryItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
class RiskDistributionBucket(DocuGuardSchema):
    label: str
    count: int
    percentage: float = 0.0


class DashboardStats(DocuGuardSchema):
    """Aggregated statistics displayed on the dashboard."""

    total_documents: int = 0
    total_analyses: int = 0
    total_markers: int = 0
    average_risk_score: float = 0.0
    decision_distribution: Dict[str, int] = Field(default_factory=dict)
    document_type_distribution: Dict[str, int] = Field(default_factory=dict)
    severity_distribution: SeverityDistribution = Field(default_factory=SeverityDistribution)
    risk_distribution: List[RiskDistributionBucket] = Field(default_factory=list)
    recent_analyses: List[HistoryItem] = Field(default_factory=list)
    model_performance: List[ModelPerformance] = Field(default_factory=list)
    files_this_week: int = 0
    analyses_this_week: int = 0
    top_marker: str = ""
    top_marker_count: int = 0
    marker_frequency: Dict[str, int] = Field(default_factory=dict)
    average_model_accuracy: Optional[float] = None


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
class ComparisonSummary(DocuGuardSchema):
    risk_difference: float = 0.0
    common_markers: List[str] = Field(default_factory=list)
    decision_matches: bool = False
    most_similar_metric: str = ""
    verdict: str = ""


class ComparedField(DocuGuardSchema):
    """Side-by-side values for a single comparable attribute."""

    attribute: str
    value_a: str = ""
    value_b: str = ""
    matches: bool = False


class CompareResult(DocuGuardSchema):
    """Side-by-side comparison of two analysed documents."""

    document_a: DocumentInfo
    document_b: DocumentInfo
    analysis_a: Optional[AnalysisResult] = None
    analysis_b: Optional[AnalysisResult] = None
    comparison: ComparisonSummary
    text_similarity: float = 0.0
    common_names: List[str] = Field(default_factory=list)
    common_dates: List[str] = Field(default_factory=list)
    common_numbers: List[str] = Field(default_factory=list)
    common_ids: List[str] = Field(default_factory=list)
    field_comparison: List[ComparedField] = Field(default_factory=list)
    differences: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Batch processing
# --------------------------------------------------------------------------- #
class BatchItemResult(DocuGuardSchema):
    original_filename: str
    status: str  # "success" | "failed"
    document_id: Optional[int] = None
    analysis_id: Optional[int] = None
    risk_score: Optional[float] = None
    decision: Optional[AnalysisDecision] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None


class CompareRequest(DocuGuardSchema):
    document_id_a: int
    document_id_b: int


class BatchAnalyzeRequest(DocuGuardSchema):
    document_ids: List[int]


class BatchResult(DocuGuardSchema):
    total_files: int
    successful: int
    failed: int
    results: List[BatchItemResult] = Field(default_factory=list)
    overall_risk_score: float = 0.0


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
class ReportRequest(DocuGuardSchema):
    """Request payload for generating a forensic report."""

    document_id: Optional[int] = None
    analysis_id: Optional[int] = None
    format: ReportFormat = ReportFormat.PDF
    include_sections: List[str] = Field(
        default_factory=lambda: [
            "summary",
            "markers",
            "ml_predictions",
            "metadata",
            "recommendations",
        ]
    )
    report_name: Optional[str] = None

    @model_validator(mode="after")
    def _at_least_one_id(self):
        if self.document_id is None and self.analysis_id is None:
            raise ValueError("Either document_id or analysis_id must be provided")
        return self


class ReportResponse(DocuGuardSchema):
    message: str = "Report generated successfully"
    report_path: str
    report_url: Optional[str] = None
    analysis_id: int
    format: ReportFormat
    generated_at: datetime
    page_count: Optional[int] = None


# --------------------------------------------------------------------------- #
# Generic API responses
# --------------------------------------------------------------------------- #
class HealthResponse(DocuGuardSchema):
    status: str = "healthy"
    version: str
    database: str = "connected"
    uptime_seconds: float = 0.0


class MessageResponse(DocuGuardSchema):
    message: str
    detail: Optional[str] = None


class ErrorResponse(DocuGuardSchema):
    error: str
    detail: Optional[str] = None
    status_code: int


class PaginationMeta(DocuGuardSchema):
    page: int = 1
    per_page: int = 20
    total_pages: int = 1
    total_items: int = 0


class PaginatedResponse(DocuGuardSchema):
    items: List[Any] = Field(default_factory=list)
    pagination: PaginationMeta = Field(default_factory=PaginationMeta)