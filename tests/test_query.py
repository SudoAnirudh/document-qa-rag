import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.models.schemas import SourceChunk
from app.core.exceptions import ExternalServiceError

client = TestClient(app)


def test_query_grounded_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POST /query with valid context returns grounded=True and sources."""
    mock_sources = [
        SourceChunk(
            document_id="doc_123",
            chunk_index=0,
            text="Retrieval-Augmented Generation enhances LLM responses.",
            score=0.85
        )
    ]

    monkeypatch.setattr(
        "app.services.retrieval.RetrievalService.retrieve",
        MagicMock(return_value=mock_sources)
    )
    monkeypatch.setattr(
        "app.services.generation.GenerationService.generate_answer",
        MagicMock(return_value="Retrieval-Augmented Generation enhances LLM responses.")
    )

    payload = {"question": "What is RAG?", "top_k": 5}
    response = client.post("/query", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is True
    assert data["answer"] == "Retrieval-Augmented Generation enhances LLM responses."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["document_id"] == "doc_123"


def test_query_unsupported_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POST /query below similarity threshold skips generation and returns grounded=False."""
    mock_sources = [
        SourceChunk(
            document_id="doc_123",
            chunk_index=0,
            text="Unrelated text content.",
            score=0.20  # Below 0.35 threshold
        )
    ]

    mock_retrieve = MagicMock(return_value=mock_sources)
    mock_generate = MagicMock()

    monkeypatch.setattr("app.services.retrieval.RetrievalService.retrieve", mock_retrieve)
    monkeypatch.setattr("app.services.generation.GenerationService.generate_answer", mock_generate)

    payload = {"question": "What is the capital of Australia?"}
    response = client.post("/query", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is False
    assert data["answer"] == "not found in the provided documents"
    assert data["sources"] == []

    # Generator MUST NOT be called when threshold fails
    mock_generate.assert_not_called()


def test_query_empty_sources_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POST /query with 0 retrieved sources returns grounded=False."""
    monkeypatch.setattr("app.services.retrieval.RetrievalService.retrieve", MagicMock(return_value=[]))
    mock_generate = MagicMock()
    monkeypatch.setattr("app.services.generation.GenerationService.generate_answer", mock_generate)

    payload = {"question": "Unknown topic?"}
    response = client.post("/query", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is False
    assert data["answer"] == "not found in the provided documents"
    assert data["sources"] == []
    mock_generate.assert_not_called()


def test_query_validation_errors() -> None:
    """Verify FastAPI Pydantic schema validation returns HTTP 422 for invalid query requests."""
    # Empty question
    resp1 = client.post("/query", json={"question": ""})
    assert resp1.status_code == 422

    # Question > 2000 chars
    resp2 = client.post("/query", json={"question": "x" * 2001})
    assert resp2.status_code == 422

    # top_k < 1
    resp3 = client.post("/query", json={"question": "Valid?", "top_k": 0})
    assert resp3.status_code == 422

    # top_k > 20
    resp4 = client.post("/query", json={"question": "Valid?", "top_k": 25})
    assert resp4.status_code == 422


def test_query_external_service_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POST /query returns HTTP 502 when retrieval/external service fails."""
    def mock_failed_retrieve(*args, **kwargs):
        raise ExternalServiceError("Embedding service temporarily unavailable")

    monkeypatch.setattr("app.services.retrieval.RetrievalService.retrieve", mock_failed_retrieve)

    payload = {"question": "What is RAG?"}
    response = client.post("/query", json=payload)
    assert response.status_code == 502
    assert "Embedding service temporarily unavailable" in response.json()["detail"]
