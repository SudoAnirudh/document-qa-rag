from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.core.exceptions import InvalidFileError, ExternalServiceError

client = TestClient(app)


def test_invalid_file_error_handler_mapping(monkeypatch) -> None:
    """Verify InvalidFileError is converted to HTTP 400 Bad Request."""
    def mock_invalid_process(*args, **kwargs):
        raise InvalidFileError("Custom parsing exception message")

    monkeypatch.setattr("app.services.ingestion.IngestionService.process_document", mock_invalid_process)

    files = {"file": ("test.txt", b"Some data", "text/plain")}
    response = client.post("/upload", files=files)

    assert response.status_code == 400
    assert response.json() == {"detail": "Custom parsing exception message"}


def test_external_service_error_handler_mapping(monkeypatch) -> None:
    """Verify ExternalServiceError is converted to HTTP 502 Bad Gateway with safe detail."""
    def mock_failed_embed(*args, **kwargs):
        raise ExternalServiceError("Embedding service temporarily unavailable")

    monkeypatch.setattr("app.services.embeddings.EmbeddingService.embed_documents", mock_failed_embed)

    files = {"file": ("valid.txt", b"Valid document text.", "text/plain")}
    response = client.post("/upload", files=files)

    assert response.status_code == 502
    assert response.json() == {"detail": "Embedding service temporarily unavailable"}
    # Verify no raw API keys or internal stack trace details are exposed
    assert "api_key" not in response.text.lower()
    assert "traceback" not in response.text.lower()
