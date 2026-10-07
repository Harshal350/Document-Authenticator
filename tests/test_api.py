"""API integration tests for DocuGuard.

Exercises the FastAPI application through its TestClient: health check, file
upload, forensic analysis, history listing and dashboard statistics. A
throw-away SQLite database plus temp upload / report directories are created
before ``app.main`` is imported so the real development database is never
touched. Every artifact lives under one temp folder that is removed on exit.
"""

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="docuguard_test_api_"))
os.environ["DATABASE_URL"] = "sqlite:///{}/test.db".format(_TEST_ROOT)
os.environ["TEMP_DIR"] = str(_TEST_ROOT / "raw")
os.environ["PROCESSED_DIR"] = str(_TEST_ROOT / "processed")
os.environ["REPORT_DIR"] = str(_TEST_ROOT / "reports")
os.environ["ALLOWED_EXTENSIONS"] = '["pdf", "txt"]'

atexit.register(lambda: shutil.rmtree(_TEST_ROOT, ignore_errors=True))

sys.path.insert(0, "backend")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

_SAMPLE_BYTES = (
    "DocuGuard API invoice\n"
    "INVOICE NUMBER 8899001122\n"
    "Date: 05/03/2024\n"
    "Subtotal: 200.00 USD\n"
    "Tax: 18.00 USD\n"
    "Total: 218.00 USD\n"
    "Customer: Northwind Traders\n"
    "This is generated purely for the automated test suite."
    .encode("utf-8")
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _upload(client, filename="sample.txt"):
    response = client.post(
        "/api/v1/upload",
        files={"file": (filename, _SAMPLE_BYTES, "text/plain")},
    )
    assert response.status_code == 200, response.text
    return response.json()["document_id"]


def _analyze(client, document_id):
    response = client.post("/api/v1/analyze/{0}".format(document_id))
    assert response.status_code == 200, response.text
    return response.json()


def _upload_and_analyze(client):
    return _analyze(client, _upload(client))


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
class TestHealth:
    def test_health_endpoint(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"] == "connected"
        assert data["version"]
        assert data["uptime_seconds"] >= 0

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["app"] == "DocuGuard"
        assert data["health"] == "/api/v1/health"


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #
class TestUpload:
    def test_upload_document(self, client):
        response = client.post(
            "/api/v1/upload",
            files={"file": ("sample.txt", _SAMPLE_BYTES, "text/plain")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] > 0
        assert data["filename"] == "sample.txt"
        assert data["message"]

    def test_upload_invalid_extension(self, client):
        response = client.post(
            "/api/v1/upload",
            files={"file": ("malware.exe", b"MZ fake payload", "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "not allowed" in response.json()["detail"].lower()

    def test_batch_upload(self, client):
        files = [
            ("files", ("a.txt", _SAMPLE_BYTES, "text/plain")),
            ("files", ("b.txt", _SAMPLE_BYTES, "text/plain")),
            ("files", ("bad.bin", b"\x00\x01 binary", "application/octet-stream")),
        ]
        response = client.post("/api/v1/upload/batch", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["uploaded_count"] == 2
        assert data["failed_count"] == 1
        assert data["failed_files"][0]["filename"] == "bad.bin"


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
class TestAnalysis:
    def test_analyze_document(self, client):
        document_id = _upload(client)
        analysis = _analyze(client, document_id)

        assert analysis["document_id"] == document_id
        assert analysis["decision"] in {"original", "suspicious", "forged"}
        assert 0 <= analysis["risk_score"] <= 100
        assert 0 <= analysis["confidence"] <= 1
        assert isinstance(analysis["markers"], list)
        assert isinstance(analysis["model_predictions"], list)
        assert analysis["document"]["id"] == document_id

    def test_analyze_missing_document(self, client):
        response = client.post("/api/v1/analyze/99999999")
        assert response.status_code == 404

    def test_get_analysis_results(self, client):
        analysis_id = _upload_and_analyze(client)["id"]
        response = client.get("/api/v1/results/{0}".format(analysis_id))
        assert response.status_code == 200
        assert response.json()["id"] == analysis_id


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #
class TestHistory:
    def test_history_listing(self, client):
        _upload_and_analyze(client)
        response = client.get("/api/v1/history")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert isinstance(data["items"], list)
        item = data["items"][0]
        assert item["analysis_id"] > 0
        assert item["risk_score"] >= 0
        assert item["decision"] in {"original", "suspicious", "forged"}
        assert "original_filename" in item

    def test_history_invalid_decision_filter(self, client):
        response = client.get("/api/v1/history", params={"decision": "bogus"})
        assert response.status_code == 400


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
class TestDashboard:
    def test_dashboard_stats(self, client):
        _upload_and_analyze(client)
        response = client.get("/api/v1/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert data["total_documents"] >= 1
        assert data["total_analyses"] >= 1
        assert data["total_markers"] >= 0
        assert data["average_risk_score"] >= 0
        assert "decision_distribution" in data
        assert data["severity_distribution"]["low"] >= 0
        assert isinstance(data["risk_distribution"], list)
        assert len(data["risk_distribution"]) == 3
        assert "recent_analyses" in data
        assert "files_this_week" in data


# --------------------------------------------------------------------------- #
# Batch Analysis & Document Comparison
# --------------------------------------------------------------------------- #
class TestBatchAndCompare:
    def test_batch_analyze(self, client):
        doc1 = _upload(client, "doc1.txt")
        doc2 = _upload(client, "doc2.txt")
        response = client.post(
            "/api/v1/analyze/batch", json={"document_ids": [doc1, doc2]}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 2
        assert data["successful"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["status"] == "success"

    def test_compare_documents(self, client):
        doc1 = _upload(client, "invoice_a.txt")
        doc2 = _upload(client, "invoice_b.txt")
        response = client.post(
            "/api/v1/compare",
            json={"document_id_a": doc1, "document_id_b": doc2},
        )
        assert response.status_code == 200
        data = response.json()
        assert "text_similarity" in data
        assert "comparison" in data
        assert "field_comparison" in data
        assert len(data["field_comparison"]) > 0
        assert data["comparison"]["verdict"] != ""

    def test_document_content_and_file(self, client):
        doc = _upload(client, "preview.txt")
        content_resp = client.get(f"/api/v1/documents/{doc}/content")
        assert content_resp.status_code == 200
        content_data = content_resp.json()
        assert content_data["document_id"] == doc
        assert "text" in content_data

        file_resp = client.get(f"/api/v1/documents/{doc}/file")
        assert file_resp.status_code == 200
        assert len(file_resp.content) > 0


# --------------------------------------------------------------------------- #
# Dataset Management
# --------------------------------------------------------------------------- #
class TestDatasetManagement:
    def test_dataset_info(self, client):
        response = client.get("/api/v1/models/dataset/info")
        assert response.status_code == 200
        data = response.json()
        assert data["total_rows"] > 0
        assert "class_distribution" in data

    def test_dataset_upload(self, client):
        csv_content = (
            "document_id,text,label\n"
            "D1,Certificate of Achievement,0\n"
            "D2,Fake Invoice Total Mismatch,1\n"
        ).encode("utf-8")
        files = {"file": ("test_data.csv", csv_content, "text/csv")}
        response = client.post("/api/v1/models/dataset/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["total_rows"] == 2
        assert "suggested_text_column" in data