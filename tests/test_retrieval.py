from unittest.mock import MagicMock
from app.services.retrieval import RetrievalService
from app.models.schemas import SourceChunk


def test_retrieval_service_retrieve() -> None:
    """Verify RetrievalService embeds question and returns mapped SourceChunk objects."""
    mock_embed = MagicMock()
    mock_embed.embed_query.return_value = [0.1, 0.2, 0.3]

    mock_vector_store = MagicMock()
    mock_vector_store.query_similarity.return_value = [
        {
            "document_id": "doc_999",
            "chunk_index": 2,
            "text": "Extracted context snippet.",
            "score": 0.88
        }
    ]

    service = RetrievalService(
        embedding_service=mock_embed,
        vector_store_service=mock_vector_store
    )

    sources = service.retrieve("What is RAG?", document_id="doc_999", top_k=5)

    assert len(sources) == 1
    assert isinstance(sources[0], SourceChunk)
    assert sources[0].document_id == "doc_999"
    assert sources[0].chunk_index == 2
    assert sources[0].score == 0.88

    mock_embed.embed_query.assert_called_once_with("What is RAG?")
    mock_vector_store.query_similarity.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=5,
        document_id="doc_999"
    )


def test_retrieval_service_empty_question() -> None:
    """Verify empty question returns empty list without calling dependencies."""
    mock_embed = MagicMock()
    mock_vector_store = MagicMock()

    service = RetrievalService(
        embedding_service=mock_embed,
        vector_store_service=mock_vector_store
    )

    assert service.retrieve("   ") == []
    mock_embed.embed_query.assert_not_called()
    mock_vector_store.query_similarity.assert_not_called()
