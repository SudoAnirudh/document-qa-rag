import pytest
from pydantic import ValidationError
from app.models.schemas import UploadResponse, QueryRequest, SourceChunk, QueryResponse


def test_upload_response_valid() -> None:
    """Verify UploadResponse model instantiation."""
    resp = UploadResponse(document_id="doc_123", filename="test.pdf", chunk_count=10)
    assert resp.document_id == "doc_123"
    assert resp.filename == "test.pdf"
    assert resp.chunk_count == 10


def test_query_request_valid_defaults() -> None:
    """Verify QueryRequest with minimum valid fields uses default top_k=5 and document_id=None."""
    req = QueryRequest(question="What is RAG?")
    assert req.question == "What is RAG?"
    assert req.document_id is None
    assert req.top_k == 5


def test_query_request_valid_custom() -> None:
    """Verify QueryRequest with custom document_id and top_k."""
    req = QueryRequest(question="What is RAG?", document_id="doc_456", top_k=10)
    assert req.question == "What is RAG?"
    assert req.document_id == "doc_456"
    assert req.top_k == 10


def test_query_request_empty_question_fails() -> None:
    """Verify QueryRequest raises ValidationError for empty string question."""
    with pytest.raises(ValidationError):
        QueryRequest(question="")


def test_query_request_too_long_question_fails() -> None:
    """Verify QueryRequest raises ValidationError for question > 2000 chars."""
    with pytest.raises(ValidationError):
        QueryRequest(question="a" * 2001)


def test_query_request_top_k_bounds() -> None:
    """Verify QueryRequest validates top_k bounds (1 <= top_k <= 20)."""
    with pytest.raises(ValidationError):
        QueryRequest(question="Valid question", top_k=0)

    with pytest.raises(ValidationError):
        QueryRequest(question="Valid question", top_k=21)


def test_query_response_grounded() -> None:
    """Verify QueryResponse serialization with sources and grounded status."""
    source = SourceChunk(
        document_id="doc_1",
        chunk_index=0,
        text="Sample context chunk.",
        score=0.85
    )
    resp = QueryResponse(answer="Sample answer.", sources=[source], grounded=True)
    assert resp.answer == "Sample answer."
    assert len(resp.sources) == 1
    assert resp.sources[0].score == 0.85
    assert resp.grounded is True
