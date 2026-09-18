import io
import pytest
from unittest.mock import MagicMock
from pathlib import Path
from pypdf import PdfWriter
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from app.core.exceptions import ExternalServiceError

client = TestClient(app)


@pytest.fixture
def mock_embedding_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture to mock EmbeddingService.embed_documents to return dummy vectors."""
    def mock_embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr("app.services.embeddings.EmbeddingService.embed_documents", mock_embed)


def test_upload_txt_success(mock_embedding_service: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POST /upload with valid TXT file returns UploadResponse with HTTP 200."""
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setattr(settings, "CHROMA_PATH", str(tmp_path / "chroma"))

    content = b"Sample document text for upload testing.\nSecond paragraph."
    files = {"file": ("test_doc.txt", io.BytesIO(content), "text/plain")}

    response = client.post("/upload", files=files)
    assert response.status_code == 200

    data = response.json()
    assert "document_id" in data
    assert data["filename"] == "test_doc.txt"
    assert data["chunk_count"] > 0


def test_upload_pdf_success(mock_embedding_service: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POST /upload with valid PDF file returns UploadResponse with HTTP 200."""
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setattr(settings, "CHROMA_PATH", str(tmp_path / "chroma"))

    def mock_parse_pdf(file_bytes: bytes) -> tuple[str, list[dict]]:
        return "Extracted text from PDF document page.", [{"page_number": 1, "text": "Extracted text from PDF document page."}]

    monkeypatch.setattr("app.services.ingestion.IngestionService.parse_pdf", mock_parse_pdf)

    # Create dummy PDF in memory
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    pdf_buf = io.BytesIO()
    writer.write(pdf_buf)

    files = {"file": ("sample.pdf", io.BytesIO(pdf_buf.getvalue()), "application/pdf")}

    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "sample.pdf"
    assert "document_id" in data



def test_upload_unsupported_extension() -> None:
    """Verify POST /upload with unsupported extension returns HTTP 400."""
    files = {"file": ("doc.docx", io.BytesIO(b"dummy data"), "application/vnd.openxmlformats-officedocument")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


def test_upload_empty_file() -> None:
    """Verify POST /upload with empty file returns HTTP 400."""
    files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_corrupt_pdf() -> None:
    """Verify POST /upload with corrupt PDF file returns HTTP 400."""
    files = {"file": ("bad.pdf", io.BytesIO(b"%PDF corrupt content"), "application/pdf")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "corrupt" in response.json()["detail"].lower()


def test_upload_embedding_service_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POST /upload returns HTTP 502 when embedding service fails."""
    def mock_failed_embed(self, texts: list[str]) -> list[list[float]]:
        raise ExternalServiceError("Embedding service temporarily unavailable")

    monkeypatch.setattr("app.services.embeddings.EmbeddingService.embed_documents", mock_failed_embed)

    files = {"file": ("valid.txt", io.BytesIO(b"Valid content string."), "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 502
    assert "Embedding service temporarily unavailable" in response.json()["detail"]
