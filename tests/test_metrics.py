import logging
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.models.schemas import SourceChunk

client = TestClient(app)


def test_query_metric_logging_grounded(monkeypatch, caplog) -> None:
    """Verify POST /query logs structured query_metric latency output when grounded."""
    caplog.set_level(logging.INFO)

    monkeypatch.setattr(
        "app.services.embeddings.EmbeddingService.embed_query",
        MagicMock(return_value=[0.1, 0.2, 0.3])
    )
    monkeypatch.setattr(
        "app.services.vector_store.VectorStoreService.query_similarity",
        MagicMock(return_value=[{"document_id": "d1", "chunk_index": 0, "text": "RAG snippet.", "score": 0.85}])
    )
    monkeypatch.setattr(
        "app.services.generation.GenerationService.generate_answer",
        MagicMock(return_value="RAG answer.")
    )

    response = client.post("/query", json={"question": "What is RAG?"})
    assert response.status_code == 200

    log_messages = [rec.message for rec in caplog.records]
    metric_logs = [msg for msg in log_messages if "query_metric" in msg]

    assert len(metric_logs) == 1
    log_line = metric_logs[0]
    assert "embedding_ms=" in log_line
    assert "retrieval_ms=" in log_line
    assert "generation_ms=" in log_line
    assert "total_ms=" in log_line
    assert "grounded=true" in log_line


def test_query_metric_logging_rejected(monkeypatch, caplog) -> None:
    """Verify POST /query logs query_metric with generation_ms=0.00 when grounding threshold rejects."""
    caplog.set_level(logging.INFO)

    monkeypatch.setattr(
        "app.services.embeddings.EmbeddingService.embed_query",
        MagicMock(return_value=[0.1, 0.2, 0.3])
    )
    monkeypatch.setattr(
        "app.services.vector_store.VectorStoreService.query_similarity",
        MagicMock(return_value=[{"document_id": "d1", "chunk_index": 0, "text": "Unrelated.", "score": 0.10}])
    )

    response = client.post("/query", json={"question": "Unrelated topic?"})
    assert response.status_code == 200

    metric_logs = [rec.message for rec in caplog.records if "query_metric" in rec.message]
    assert len(metric_logs) == 1
    log_line = metric_logs[0]
    assert "generation_ms=0.00" in log_line
    assert "grounding_rejected=true" in log_line
