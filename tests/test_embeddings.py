import pytest
from unittest.mock import MagicMock
from openai import APITimeoutError, RateLimitError, APIError
from app.core.exceptions import ExternalServiceError
from app.services.embeddings import EmbeddingService


def test_embed_documents_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify successful embedding call using mocked OpenAI client."""
    service = EmbeddingService(api_key="test-key")

    mock_response = MagicMock()
    mock_item1 = MagicMock()
    mock_item1.embedding = [0.1, 0.2, 0.3]
    mock_item2 = MagicMock()
    mock_item2.embedding = [0.4, 0.5, 0.6]
    mock_response.data = [mock_item1, mock_item2]

    monkeypatch.setattr(service.client.embeddings, "create", MagicMock(return_value=mock_response))

    embeddings = service.embed_documents(["chunk 1", "chunk 2"])
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]


def test_embed_query_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify single query embedding call."""
    service = EmbeddingService(api_key="test-key")

    mock_response = MagicMock()
    mock_item = MagicMock()
    mock_item.embedding = [0.9, 0.8, 0.7]
    mock_response.data = [mock_item]

    monkeypatch.setattr(service.client.embeddings, "create", MagicMock(return_value=mock_response))

    vec = service.embed_query("Sample question")
    assert vec == [0.9, 0.8, 0.7]


def test_embed_empty_input() -> None:
    """Verify empty input returns empty list without calling API."""
    service = EmbeddingService(api_key="test-key")
    assert service.embed_documents([]) == []
    assert service.embed_query("  ") == []


def test_embed_timeout_raises_external_service_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify APITimeoutError is caught and converted to ExternalServiceError."""
    service = EmbeddingService(api_key="test-key")

    mock_create = MagicMock(side_effect=APITimeoutError(request=MagicMock()))
    monkeypatch.setattr(service.client.embeddings, "create", mock_create)

    with pytest.raises(ExternalServiceError, match="Embedding service temporarily unavailable"):
        service.embed_documents(["test text"])


def test_embed_rate_limit_raises_external_service_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify RateLimitError is caught and converted to ExternalServiceError."""
    service = EmbeddingService(api_key="test-key")

    mock_create = MagicMock(side_effect=RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(status_code=429),
        body={}
    ))
    monkeypatch.setattr(service.client.embeddings, "create", mock_create)

    with pytest.raises(ExternalServiceError, match="Embedding service temporarily unavailable"):
        service.embed_documents(["test text"])


def test_embed_api_error_raises_external_service_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify general APIError is caught and converted to ExternalServiceError."""
    service = EmbeddingService(api_key="test-key")

    mock_create = MagicMock(side_effect=APIError(
        message="Internal server error from OpenAI",
        request=MagicMock(),
        body={}
    ))
    monkeypatch.setattr(service.client.embeddings, "create", mock_create)

    with pytest.raises(ExternalServiceError, match="Embedding service temporarily unavailable"):
        service.embed_query("sample question")
