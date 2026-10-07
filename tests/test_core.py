"""Core pipeline tests for DocuGuard.

Covers document processing (TXT / PDF), feature extraction, marker detection
(date / number / structural / textual), unified risk scoring and the
explainability engine. Temporary files are created inside ``tmp_path`` and are
cleaned up automatically by pytest.
"""

import sys

import pytest

sys.path.insert(0, "backend")

from app.core.document_processor import DocumentProcessor  # noqa: E402
from app.core.explainability import ExplainabilityEngine  # noqa: E402
from app.core.feature_engineering import FeatureEngineer  # noqa: E402
from app.core.marker_detector import MarkerDetector  # noqa: E402
from app.core.risk_engine import RiskEngine  # noqa: E402

SAMPLE_TEXT = (
    "INVOICE NUMBER 9843610701\n"
    "Issue Date: 12 Jan 2025\n"
    "Subtotal: 100.00 USD\n"
    "Tax: 10.00 USD\n"
    "Total: 115.00 USD\n"
    "Customer: ACME Corporation\n"
    "Payment due within 30 days of issue."
)


@pytest.fixture(scope="module")
def processor():
    return DocumentProcessor()


@pytest.fixture(scope="module")
def engineer():
    return FeatureEngineer()


@pytest.fixture(scope="module")
def detector():
    return MarkerDetector()


@pytest.fixture(scope="module")
def risk_engine():
    return RiskEngine()


@pytest.fixture(scope="module")
def explainer():
    return ExplainabilityEngine()


# --------------------------------------------------------------------------- #
# Document processing
# --------------------------------------------------------------------------- #
class TestDocumentProcessing:
    def test_txt_extraction(self, tmp_path, processor):
        content = "This is a plain text invoice for 1,200.00 dollars."
        path = tmp_path / "sample.txt"
        path.write_text(content, encoding="utf-8")

        result = processor.process_document(str(path))

        assert result["success"] is True
        assert result["error"] is None
        assert result["text"] == content
        assert result["pages"] == 1
        assert result["stats"]["file_type"] == "txt"
        assert result["stats"]["words"] == 9
        assert result["stats"]["file_size"] > 0
        assert "encoding" in result["metadata"]

    def test_pdf_extraction(self, tmp_path, processor):
        fitz = pytest.importorskip("fitz")

        document = fitz.open()
        page = document.new_page()
        text = "DocuGuard forensic test document for PDF extraction."
        page.insert_text((72, 72), text, fontsize=12, fontname="helv")
        path = tmp_path / "sample.pdf"
        document.save(str(path))
        document.close()

        result = processor.process_document(str(path))

        assert result["success"] is True
        assert text.split()[0] in result["text"]
        assert result["stats"]["file_type"] == "pdf"
        assert result["stats"]["pages"] == 1
        assert result["stats"]["words"] > 0

    def test_missing_file(self, processor):
        result = processor.process_document("/non/existent/document.pdf")
        assert result["success"] is False
        assert result["error"]
        assert result["text"] == ""

    def test_binary_rejected_as_text(self, tmp_path, processor):
        path = tmp_path / "corrupt.txt"
        path.write_bytes(b"\x00\x01\x02\x03\x04 binary payload")
        result = processor.process_document(str(path))
        assert result["success"] is False
        assert "binary" in result["error"].lower()


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
class TestFeatureExtraction:
    def test_all_feature_keys_present(self, engineer):
        features = engineer.extract_features(
            SAMPLE_TEXT,
            {"author": "ACME Corp", "creator": "DocuGuard"},
            {"pages": 1, "ocr_confidence": 0.94},
        )
        for name in engineer.FEATURE_NAMES:
            assert name in features, "missing feature {!r}".format(name)

    def test_numerical_and_date_features(self, engineer):
        text = (
            "Invoice #1234567890 dated 15/02/2024 for $1,000.00. "
            "Another charge of 250.50 EUR was added."
        )
        features = engineer.extract_features(text, {}, {})
        assert features["date_count"] >= 1
        assert features["monetary_values_count"] >= 1
        assert features["number_count"] >= 1

    def test_suspicious_phrases_flag(self, engineer):
        features = engineer.extract_features(
            "This is a certified true copy. Sample document shown for "
            "illustrative purposes only."
        )
        assert features["suspicious_phrase_count"] >= 1

    def test_metadata_features(self, engineer):
        metadata = {"author": "John Doe", "title": "Lease Agreement", "subject": "Housing"}
        features = engineer.extract_features(SAMPLE_TEXT, metadata, {})
        assert features["has_author"] == 1.0
        assert features["metadata_completeness"] > 0.0

    def test_feature_vector_shape(self, engineer):
        vector = engineer.create_feature_vector(SAMPLE_TEXT, {}, {})
        assert vector.shape == (len(engineer.FEATURE_NAMES),)


# --------------------------------------------------------------------------- #
# Marker detection
# --------------------------------------------------------------------------- #
class TestMarkerDetection:
    def test_date_markers(self, detector):
        text = "Renewal on 12/31/2099 and the earlier record on 02/30/2025."
        markers = detector.detect_all_markers(text, {}, {})
        names = {m["name"] for m in markers}
        assert "impossible_dates" in names
        assert "future_dates" in names

    def test_number_markers(self, detector):
        text = (
            "Subtotal: 100.00 USD. Tax: 10.00 USD. Total: 115.00 USD. "
            "REF 88887766 88887766 88887766"
        )
        markers = detector.detect_all_markers(text, {}, {})
        names = {m["name"] for m in markers}
        assert "inconsistent_totals" in names
        assert "repeated_numbers" in names

    def test_structural_markers(self, detector):
        text = (
            "This is a contract between the buyer and the seller. "
            "The goods are described in the schedule attached."
        )
        markers = detector.detect_all_markers(text, {}, {"page_count": 4})
        names = {m["name"] for m in markers}
        assert "missing_sections" in names
        assert any(m["category"] == "structural" for m in markers)

    def test_textual_suspicious_phrases(self, detector):
        text = "This is a certified true copy. For illustrative purposes only."
        markers = detector.detect_all_markers(text, {}, {})
        names = {m["name"] for m in markers}
        assert "suspicious_phrases" in names

    def test_sorted_by_severity(self, detector):
        text = (
            "Issue Date: 02/30/2025.\n"
            "This is a certified true copy.\n"
            "REF 7777777 7777777 7777777."
        )
        markers = detector.detect_all_markers(text, {}, {})
        order = {"high": 0, "medium": 1, "low": 2}
        severities = [order.get(m["severity"], 3) for m in markers]
        assert severities == sorted(severities)


# --------------------------------------------------------------------------- #
# Risk scoring
# --------------------------------------------------------------------------- #
class TestRiskEngine:
    def test_clean_document_is_original(self, risk_engine):
        result = risk_engine.calculate_risk([], [], {}, {})
        assert result["risk_score"] < 30
        assert result["decision"] == "original"
        assert result["ml_risk_score"] == 50.0

    def test_forged_document_high_risk(self, risk_engine):
        predictions = [
            {
                "model_name": "random_forest",
                "genuine_probability": 0.05,
                "fake_probability": 0.95,
            }
        ]
        markers = [
            {"category": "date", "name": "impossible_dates", "severity": "high", "score": 0.9},
            {"category": "numerical", "name": "inconsistent_totals", "severity": "high", "score": 0.9},
            {"category": "structural", "name": "missing_sections", "severity": "high", "score": 0.9},
            {"category": "id_reference", "name": "repeated_ids", "severity": "high", "score": 0.9},
        ]
        result = risk_engine.calculate_risk(predictions, markers, {}, {})
        assert result["risk_score"] >= 60
        assert result["decision"] == "forged"
        assert 0.0 <= result["confidence"] <= 1.0

    def test_category_decision_bands(self, risk_engine):
        cases = [
            (0.0, "original"),
            (29.0, "original"),
            (30.0, "suspicious"),
            (59.0, "suspicious"),
            (60.0, "forged"),
            (100.0, "forged"),
        ]
        for score, expected in cases:
            assert risk_engine._decision(score) == expected

    def test_medium_band_using_ml_only(self, risk_engine):
        result = risk_engine.calculate_risk(
            [{"model_name": "m", "fake_probability": 1.0}], [], {}, {}
        )
        assert 30.0 <= result["risk_score"] < 60.0
        assert result["decision"] == "suspicious"

    def test_marker_score_normalization(self, risk_engine):
        markers_01 = [
            {"category": "date", "name": "m", "severity": "high", "score": 0.9}
        ]
        markers_100 = [
            {"category": "date", "name": "m", "severity": "high", "score": 90}
        ]
        r1 = risk_engine.calculate_risk([], markers_01, {}, {})
        r2 = risk_engine.calculate_risk([], markers_100, {}, {})
        assert r1["marker_risk_score"] == r2["marker_risk_score"] == 45.0

    def test_all_result_keys_present(self, risk_engine):
        result = risk_engine.calculate_risk([], [], {}, {})
        expected = {
            "risk_score",
            "decision",
            "confidence",
            "ml_risk_score",
            "marker_risk_score",
            "structural_risk_score",
            "consistency_risk_score",
        }
        assert expected.issubset(result.keys())


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
class TestExplainability:
    def test_output_format(self, explainer):
        result = explainer.explain(
            risk_score=66.86,
            decision="forged",
            model_predictions=[
                {
                    "model_name": "random_forest",
                    "genuine_probability": 0.05,
                    "fake_probability": 0.95,
                }
            ],
            markers=[
                {
                    "category": "date",
                    "name": "impossible_dates",
                    "severity": "high",
                    "score": 0.9,
                }
            ],
            feature_dict={},
            stats={},
        )
        assert {
            "summary",
            "top_factors",
            "risk_breakdown",
            "marker_summary",
            "ml_summary",
        }.issubset(result.keys())

        assert isinstance(result["summary"], str) and result["summary"]
        assert isinstance(result["top_factors"], list)
        assert result["top_factors"]

        breakdown = result["risk_breakdown"]
        assert breakdown["decision"] == "forged"
        assert 0.0 <= breakdown["risk_score"] <= 100.0
        assert "decision_label" in breakdown
        assert "ml_risk_score" in breakdown

    def test_no_markers_summary(self, explainer):
        result = explainer.explain(
            risk_score=10.0,
            decision="original",
            model_predictions=[],
            markers=[],
            feature_dict={},
            stats={},
        )
        assert (
            result["marker_summary"]
            == "No forgery indicators were detected during marker analysis."
        )
        assert "neutral" in result["ml_summary"].lower()

    def test_summary_references_risk_score(self, explainer):
        result = explainer.explain(
            risk_score=72.5,
            decision="forged",
            model_predictions=[
                {
                    "model_name": "xgboost",
                    "genuine_probability": 0.2,
                    "fake_probability": 0.8,
                }
            ],
            markers=[
                {
                    "category": "textual",
                    "name": "suspicious_phrases",
                    "severity": "medium",
                    "score": 0.7,
                }
            ],
            feature_dict={},
            stats={},
        )
        assert "72.5" in result["summary"]
        assert result["risk_breakdown"]["model_count"] == 1
        assert result["risk_breakdown"]["marker_count"] == 1