from pathlib import Path
from app.services.chunking import DocumentChunk
from app.services.vector_store import VectorStoreService


def test_vector_store_add_and_query(tmp_path: Path) -> None:
    """Verify adding document chunks to persistent Chroma vector store and querying similarity."""
    store = VectorStoreService(chroma_path=str(tmp_path / "chroma_test"))

    chunks = [
        DocumentChunk(
            document_id="doc_A",
            filename="doc_A.txt",
            chunk_index=0,
            text="Python fast API web framework.",
            metadata={"document_id": "doc_A", "filename": "doc_A.txt", "chunk_index": 0}
        ),
        DocumentChunk(
            document_id="doc_A",
            filename="doc_A.txt",
            chunk_index=1,
            text="Vector search with Chroma DB.",
            metadata={"document_id": "doc_A", "filename": "doc_A.txt", "chunk_index": 1}
        ),
        DocumentChunk(
            document_id="doc_B",
            filename="doc_B.txt",
            chunk_index=0,
            text="Unrelated recipe for French omelette.",
            metadata={"document_id": "doc_B", "filename": "doc_B.txt", "chunk_index": 0}
        ),
    ]

    # Dummy 3-dimensional embeddings for testing
    embeddings = [
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
        [0.0, 0.0, 1.0],
    ]

    added_count = store.add_chunks(chunks, embeddings)
    assert added_count == 3

    # Query with embedding close to doc_A
    query_vec = [0.95, 0.05, 0.0]
    results = store.query_similarity(query_vec, top_k=2)

    assert len(results) == 2
    assert results[0]["document_id"] == "doc_A"
    assert results[0]["score"] > 0.5


def test_vector_store_document_id_filter(tmp_path: Path) -> None:
    """Verify document_id metadata filter restricts search results strictly to target document."""
    store = VectorStoreService(chroma_path=str(tmp_path / "chroma_test_filter"))

    chunks = [
        DocumentChunk(
            document_id="doc_1",
            filename="doc_1.txt",
            chunk_index=0,
            text="Machine learning RAG system.",
            metadata={"document_id": "doc_1", "filename": "doc_1.txt", "chunk_index": 0}
        ),
        DocumentChunk(
            document_id="doc_2",
            filename="doc_2.txt",
            chunk_index=0,
            text="Machine learning RAG system copy.",
            metadata={"document_id": "doc_2", "filename": "doc_2.txt", "chunk_index": 0}
        ),
    ]

    embeddings = [
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ]

    store.add_chunks(chunks, embeddings)

    # Filter for doc_1 specifically
    results_doc1 = store.query_similarity(query_embedding=[0.5, 0.5, 0.5], top_k=5, document_id="doc_1")
    assert len(results_doc1) == 1
    assert results_doc1[0]["document_id"] == "doc_1"


def test_vector_store_empty_collection(tmp_path: Path) -> None:
    """Verify querying empty store returns empty list."""
    store = VectorStoreService(chroma_path=str(tmp_path / "chroma_empty"))
    results = store.query_similarity(query_embedding=[0.1, 0.2, 0.3], top_k=5)
    assert results == []
