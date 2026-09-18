from app.services.chunking import ChunkingService


def test_chunking_default_config() -> None:
    """Verify chunking document with default chunk size and overlap."""
    service = ChunkingService()
    sample_text = "Paragraph one text. " * 30 + "\n\n" + "Paragraph two text. " * 30

    chunks = service.chunk_document(
        text=sample_text,
        document_id="doc_test_123",
        filename="test_doc.txt"
    )

    assert len(chunks) > 1
    assert chunks[0].document_id == "doc_test_123"
    assert chunks[0].filename == "test_doc.txt"
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert "document_id" in chunks[0].metadata
    assert "filename" in chunks[0].metadata
    assert "chunk_index" in chunks[0].metadata


def test_chunking_custom_config() -> None:
    """Verify chunking respects custom chunk_size and chunk_overlap settings."""
    small_chunk_service = ChunkingService(chunk_size=100, chunk_overlap=20)
    large_chunk_service = ChunkingService(chunk_size=500, chunk_overlap=50)

    sample_text = "Word " * 150  # ~750 characters

    small_chunks = small_chunk_service.chunk_document(
        text=sample_text,
        document_id="doc_custom",
        filename="sample.txt"
    )
    large_chunks = large_chunk_service.chunk_document(
        text=sample_text,
        document_id="doc_custom",
        filename="sample.txt"
    )

    assert len(small_chunks) > len(large_chunks)
    assert small_chunk_service.chunk_size == 100
    assert small_chunk_service.chunk_overlap == 20


def test_chunking_empty_text() -> None:
    """Verify empty or whitespace text returns empty list of chunks."""
    service = ChunkingService()
    chunks = service.chunk_document(
        text="   \n\t ",
        document_id="doc_empty",
        filename="empty.txt"
    )
    assert chunks == []


def test_chunking_metadata_indexing() -> None:
    """Verify sequential chunk index assignments and metadata dictionary structure."""
    service = ChunkingService(chunk_size=200, chunk_overlap=20)
    text = "Section 1: Introduction.\n" + "Details... " * 40

    chunks = service.chunk_document(
        text=text,
        document_id="doc_seq",
        filename="report.pdf"
    )

    for idx, chunk in enumerate(chunks):
        assert chunk.chunk_index == idx
        assert chunk.metadata["chunk_index"] == idx
        assert chunk.metadata["document_id"] == "doc_seq"
        assert chunk.metadata["filename"] == "report.pdf"
