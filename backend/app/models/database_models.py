"""SQLAlchemy ORM models for the DocuGuard application.

Tables
------
``documents``        - metadata about every uploaded file.
``analyses``         - one forensic analysis run per analysed document.
``markers``          - individual findings raised during an analysis.
``model_predictions``- per-model ML predictions for a given analysis.
``model_metrics``    - evaluation metrics for trained/registered ML models.
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.database import Base


class Document(Base):
    """A file that has been uploaded to DocuGuard."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    words: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    characters: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tables_detected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ocr_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    document_type: Mapped[str] = mapped_column(String(20), nullable=False, default="original")
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    analyses: Mapped[List["Analysis"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="Analysis.created_at.desc()",
    )

    __table_args__ = (
        Index("ix_documents_uploaded_at", "uploaded_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Document id={self.id} filename={self.filename!r} type={self.file_type!r}>"


class Analysis(Base):
    """A complete forensic analysis run against a single document."""

    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False
    )
    risk_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    decision: Mapped[str] = mapped_column(String(50), index=True, nullable=False, default="original")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    ml_risk_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    marker_risk_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    structural_risk_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    consistency_risk_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    document: Mapped["Document"] = relationship(back_populates="analyses")
    markers: Mapped[List["Marker"]] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
        order_by="Marker.score.desc()",
    )
    model_predictions: Mapped[List["ModelPrediction"]] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_analyses_created_at", "created_at"),
        Index("ix_analyses_decision_created", "decision", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Analysis id={self.id} document_id={self.document_id} decision={self.decision!r}>"


class Marker(Base):
    """A single forensic finding detected during an analysis."""

    __tablename__ = "markers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True, nullable=False
    )
    category: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), index=True, nullable=False, default="medium")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    evidence: Mapped[str] = mapped_column(Text, nullable=False, default="")
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    analysis: Mapped["Analysis"] = relationship(back_populates="markers")

    __table_args__ = (
        Index("ix_markers_category_severity", "category", "severity"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Marker id={self.id} analysis_id={self.analysis_id} name={self.name!r}>"


class ModelPrediction(Base):
    """Machine-learning prediction contributed by a single model."""

    __tablename__ = "model_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True, nullable=False
    )
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)
    genuine_probability: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fake_probability: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    analysis: Mapped["Analysis"] = relationship(back_populates="model_predictions")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<ModelPrediction id={self.id} model={self.model_name!r}>"


class ModelMetric(Base):
    """Evaluation metrics recorded for a trained or registered model."""

    __tablename__ = "model_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    precision_: Mapped[float] = mapped_column("precision", Float, nullable=False, default=0.0)
    recall: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    f1: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    roc_auc: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    training_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<ModelMetric model={self.model_name!r} accuracy={self.accuracy:.3f}>"