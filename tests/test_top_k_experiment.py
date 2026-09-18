from pathlib import Path
from app.services.chunking import ChunkingService
from evaluation.evaluate import run_top_k_experiment


def test_top_k_experiment_execution() -> None:
    """Verify that run_top_k_experiment evaluates top_k=3, 5, and 8 correctly."""
    doc_path = Path(__file__).parent.parent / "evaluation" / "test_document.txt"
    chunker = ChunkingService(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_document(
        doc_path.read_text(encoding="utf-8"),
        document_id="eval_doc_01",
        filename="test_document.txt"
    )

    results = run_top_k_experiment(chunks, [])

    assert "top_k=3" in results
    assert "top_k=5" in results
    assert "top_k=8" in results

    assert results["top_k=3"]["chunks_retrieved"] == 3
    assert results["top_k=5"]["chunks_retrieved"] == 5
    assert results["top_k=8"]["chunks_retrieved"] == 8

    # Verify context char count increases with top_k
    assert results["top_k=3"]["total_context_chars"] < results["top_k=5"]["total_context_chars"]
    assert results["top_k=5"]["total_context_chars"] < results["top_k=8"]["total_context_chars"]
