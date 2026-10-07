"""Report generation endpoints for the DocuGuard API.

A professional, single-file PDF report is produced for a completed analysis
using ``reportlab``. The report covers document information, the risk
assessment, per-model predictions, detected markers, an evidence summary
(including the explainability payload stored at analysis time) and a standard
disclaimer. Reports are written into the configured ``reports`` directory and
served back as static files.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.api.analysis import _load_explanation, _map_marker_category, _map_severity
from app.config import get_settings
from app.database.database import get_db
from app.models.database_models import Analysis
from app.models.schemas import ReportFormat, ReportResponse
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter()

DISCLAIMER = (
    "This report represents an AI-assisted risk assessment based on document "
    "characteristics, extracted content, structural properties and trained "
    "machine learning models. It does not constitute legal authentication or "
    "definitive proof that a document is genuine or fraudulent."
)

DECISION_LABELS = {
    "original": "Likely Genuine",
    "suspicious": "Likely Fake/Suspicious",
    "forged": "Likely Fake",
}

DOCUMENT_TITLE = "DocuGuard Analysis Report"


def _decision_label(decision: str) -> str:
    """Human-readable label for a stored analysis decision."""
    return DECISION_LABELS.get(str(decision).lower(), str(decision))


def _fmt_bytes(size: Any) -> str:
    """Format a byte count into a readable string."""
    try:
        size = int(size)
    except (TypeError, ValueError):
        return "0 B"
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.2f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


def _safe_text(value: Any, fallback: str = "-") -> str:
    """Coerce a value into a safe display string."""
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _explanation_text(
    explanation: Optional[Dict[str, Any]], analysis: Analysis
) -> str:
    """Flatten the stored explainability payload into a summary paragraph."""
    if explanation is not None:
        summary = explanation.get("summary")
        if summary:
            return str(summary)
        lines = []
        for key, value in explanation.items():
            if isinstance(value, (dict, list)):
                continue
            lines.append(f"{str(key).replace('_', ' ').title()}: {value}")
        if lines:
            return "; ".join(lines)

    return (
        f"Analysis #{analysis.id} produced a risk score of "
        f"{analysis.risk_score:.1f}/100 ({_decision_label(analysis.decision)}) "
        f"with {len(analysis.markers)} detected markers and "
        f"{len(analysis.model_predictions)} model predictions."
    )


def _pdf_page_count(path: Path) -> Optional[int]:
    """Return the generated PDF's page count when PyMuPDF is available."""
    try:
        import fitz

        with fitz.open(str(path)) as pdf:
            return pdf.page_count
    except Exception:
        return None


def _build_styles() -> Dict[str, ParagraphStyle]:
    """Create the paragraph styles used throughout the report."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            name="DocuTitle",
            parent=base["Title"],
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=6,
            textColor=colors.HexColor("#1f3a5f"),
        ),
        "subtitle": ParagraphStyle(
            name="DocuSubtitle",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#5b6b7f"),
            spaceAfter=18,
        ),
        "h2": ParagraphStyle(
            name="DocuHeading",
            parent=base["Heading2"],
            fontSize=14,
            leading=17,
            spaceBefore=6,
            spaceAfter=8,
            textColor=colors.HexColor("#1f3a5f"),
        ),
        "body": ParagraphStyle(
            name="DocuBody",
            parent=base["Normal"],
            fontSize=9.5,
            leading=13,
            alignment=TA_JUSTIFY,
        ),
        "small": ParagraphStyle(
            name="DocuSmall",
            parent=base["Normal"],
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#3a4a5a"),
        ),
        "disclaimer": ParagraphStyle(
            name="DocuDisclaimer",
            parent=base["Italic"],
            fontSize=8.5,
            leading=11,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#6d6d6d"),
        ),
    }


def _kv_table(rows: List[List[Any]], styles: Dict[str, ParagraphStyle]) -> Table:
    """Build a two-column key/value table from ``rows``."""
    data: List[List[Any]] = [["Field", "Value"]]
    data.extend(
        [
            [
                Paragraph(_safe_text(key), styles["small"]),
                Paragraph(_safe_text(value), styles["body"]),
            ]
            for key, value in rows
        ]
    )
    table = Table(data, colWidths=[1.6 * inch, 4.9 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#eef2f7")],
                ),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b9c4d0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _simple_table(
    header: List[str],
    rows: List[List[Any]],
    col_widths: List[float],
) -> Table:
    """Build a formatted table with the given header and cell data."""
    data: List[List[Any]] = [header]
    data.extend(rows)
    table = Table(
        data,
        colWidths=[width * inch for width in col_widths],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#eef2f7")],
                ),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b9c4d0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _generate_pdf(analysis: Analysis, report_path: Path, explanation: Optional[Dict[str, Any]]) -> int:
    """Render the full analysis report into *report_path* (returns page count)."""
    styles = _build_styles()
    document = analysis.document

    story = [
        Paragraph(DOCUMENT_TITLE, styles["title"]),
        Paragraph(
            (
                "Forensic Document Analysis | Generated "
                f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
            ),
            styles["subtitle"],
        ),
        Paragraph("1. Document Information", styles["h2"]),
        _kv_table(
            [
                ("Analysis ID", analysis.id),
                ("Document ID", document.id if document else "-"),
                ("Filename", document.original_filename if document else "-"),
                ("File Type", document.file_type if document else "-"),
                ("File Size", _fmt_bytes(document.file_size if document else 0)),
                ("Pages", document.pages if document else "-"),
                ("Words", document.words if document else "-"),
                ("Uploaded At", document.uploaded_at if document else "-"),
            ],
            styles,
        ),
        Spacer(1, 12),
        Paragraph("2. Risk Assessment", styles["h2"]),
        Paragraph(
            (
                f"Overall risk score: <b>{analysis.risk_score:.1f}/100</b><br/>"
                f"Decision: <b>{_decision_label(analysis.decision)}</b><br/>"
                f"Confidence: <b>{analysis.confidence:.1%}</b>"
            ),
            styles["body"],
        ),
        Spacer(1, 8),
        _simple_table(
            ["Component", "Score (0-100)"],
            [
                ["Machine Learning Risk", f"{analysis.ml_risk_score:.1f}"],
                ["Marker Risk", f"{analysis.marker_risk_score:.1f}"],
                ["Structural Risk", f"{analysis.structural_risk_score:.1f}"],
                ["Consistency Risk", f"{analysis.consistency_risk_score:.1f}"],
            ],
            [4.0, 2.5],
        ),
        PageBreak(),
        Paragraph("3. Model Predictions", styles["h2"]),
    ]

    predictions = analysis.model_predictions
    if predictions:
        story.append(
            _simple_table(
                ["Model", "Genuine Probability", "Fake Probability"],
                [
                    [
                        _safe_text(prediction.model_name),
                        f"{prediction.genuine_probability:.4f}",
                        f"{prediction.fake_probability:.4f}",
                    ]
                    for prediction in predictions
                ],
                [2.6, 3.1, 3.1],
            )
        )
    else:
        story.append(
            Paragraph(
                "No machine-learning predictions were recorded for this analysis.",
                styles["body"],
            )
        )

    story.append(Spacer(1, 16))
    story.append(Paragraph("4. Detected Markers", styles["h2"]))

    markers = analysis.markers
    if markers:
        story.append(
            _simple_table(
                ["Category", "Name", "Severity", "Score", "Description"],
                [
                    [
                        _safe_text(
                            _map_marker_category(marker.category).value.title()
                        ),
                        _safe_text(marker.name),
                        _safe_text(_map_severity(marker.severity).value.title()),
                        f"{marker.score:.1f}",
                        _safe_text(marker.description)[:160],
                    ]
                    for marker in markers
                ],
                [1.1, 1.6, 0.9, 0.7, 2.2],
            )
        )
    else:
        story.append(
            Paragraph("No markers were detected during the analysis.", styles["body"])
        )

    story.append(PageBreak())
    story.append(Paragraph("5. Evidence Summary", styles["h2"]))
    story.append(Paragraph(_explanation_text(explanation, analysis), styles["body"]))

    story.append(Spacer(1, 16))
    for marker in markers:
        evidence = _safe_text(marker.evidence)
        if evidence:
            story.append(
                Paragraph(
                    f"<b>{_safe_text(marker.name)}</b> (severity "
                    f"{_safe_text(_map_severity(marker.severity).value.title())}, "
                    f"score {marker.score:.1f}): {evidence}",
                    styles["body"],
                )
            )

    story.extend(
        [
            Spacer(1, 24),
            Paragraph("6. Disclaimer", styles["h2"]),
            Paragraph(DISCLAIMER, styles["disclaimer"]),
        ]
    )

    SimpleDocTemplate(
        str(report_path),
        pagesize=letter,
        title=DOCUMENT_TITLE,
        author="DocuGuard",
        subject=f"Forensic analysis report for {_decision_label(analysis.decision)}",
    ).build(story)
    return _pdf_page_count(report_path) or 0


def _report_filename(analysis_id: int) -> str:
    """Generate a unique, timestamped filename for a report PDF."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"report_{analysis_id}_{timestamp}.pdf"


@router.get("/report/{analysis_id}", response_model=ReportResponse)
async def generate_report(
    analysis_id: int,
    db: Session = Depends(get_db),
) -> ReportResponse:
    """Generate a PDF forensic report for a completed analysis.

    The report is written to the configured reports directory (served back as
    static files under ``/api/v1/reports``) and its path / URL are returned.
    """
    analysis = db.execute(
        select(Analysis)
        .where(Analysis.id == analysis_id)
        .options(
            selectinload(Analysis.document),
            selectinload(Analysis.markers),
            selectinload(Analysis.model_predictions),
        )
    ).scalar_one_or_none()
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    report_dir = Path(settings.report_dir)
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Could not create reports directory %s: %s", report_dir, exc)
        raise HTTPException(
            status_code=500, detail="Reports directory is not available"
        )

    report_path = report_dir / _report_filename(analysis_id)
    try:
        page_count = _generate_pdf(
            analysis, report_path, _load_explanation(analysis_id)
        )
    except Exception as exc:
        logger.exception("PDF generation failed for analysis #%s", analysis_id)
        report_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500, detail=f"Report generation failed: {exc}"
        )

    logger.info(
        "Generated report %s for analysis #%s (%d pages)",
        report_path.name,
        analysis_id,
        page_count,
    )
    return ReportResponse(
        message="Report generated successfully",
        report_path=str(report_path),
        report_url=f"{settings.api_prefix}/reports/{report_path.name}",
        analysis_id=analysis_id,
        format=ReportFormat.PDF,
        generated_at=datetime.utcnow(),
        page_count=page_count if page_count else None,
    )