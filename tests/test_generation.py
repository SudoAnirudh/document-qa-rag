import pytest
from unittest.mock import MagicMock
from openai import APITimeoutError, RateLimitError, APIError
from app.core.exceptions import ExternalServiceError
from app.models.schemas import SourceChunk
from app.services.generation import GenerationService


def test_generation_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify successful grounded answer generation using mocked OpenAI client."""
    service = GenerationService(api_key="test-key")

    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Retrieval-Augmented Generation enhances LLM responses."
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    monkeypatch.setattr(service.client.chat.completions, "create", MagicMock(return_value=mock_response))

    chunk = SourceChunk(
        document_id="doc_1",
        chunk_index=0,
        text="Retrieval-Augmented Generation enhances LLM responses.",
        score=0.85
    )

    answer = service.generate_answer("What is RAG?", [chunk])
    assert answer == "Retrieval-Augmented Generation enhances LLM responses."


def test_generation_empty_context() -> None:
    """Verify empty context chunks list returns 'not found in the provided documents' without calling API."""
    service = GenerationService(api_key="test-key")
    answer = service.generate_answer("What is RAG?", [])
    assert answer == "not found in the provided documents"


def test_generation_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify APITimeoutError is caught and converted to ExternalServiceError."""
    service = GenerationService(api_key="test-key")
    mock_create = MagicMock(side_effect=APITimeoutError(request=MagicMock()))
    monkeypatch.setattr(service.client.chat.completions, "create", mock_create)

    chunk = SourceChunk(document_id="d1", chunk_index=0, text="sample text", score=0.9)

    with pytest.raises(ExternalServiceError, match="Generation service temporarily unavailable"):
        service.generate_answer("question", [chunk])


def test_generation_rate_limit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify RateLimitError is caught and converted to ExternalServiceError."""
    service = GenerationService(api_key="test-key")
    mock_create = MagicMock(side_effect=RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(status_code=429),
        body={}
    ))
    monkeypatch.setattr(service.client.chat.completions, "create", mock_create)

    chunk = SourceChunk(document_id="d1", chunk_index=0, text="sample text", score=0.9)

    with pytest.raises(ExternalServiceError, match="Generation service temporarily unavailable"):
        service.generate_answer("question", [chunk])
