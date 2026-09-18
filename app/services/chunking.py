from dataclasses import dataclass
from typing import Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.core.logging import logger


@dataclass
class DocumentChunk:
    """Dataclass representing an isolated, metadata-tagged document chunk."""

    document_id: str
    filename: str
    chunk_index: int
    text: str
    metadata: dict[str, Any]


class ChunkingService:
    """Isolated utility service wrapping RecursiveCharacterTextSplitter for document chunking."""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        self.chunk_size = chunk_size if chunk_size is not None else settings.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.DEFAULT_CHUNK_OVERLAP

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_document(
        self,
        text: str,
        document_id: str,
        filename: str,
        pages_data: list[dict[str, Any]] | None = None
    ) -> list[DocumentChunk]:
        """Split raw text into structured DocumentChunk objects with metadata.

        Args:
            text: Raw extracted full text of the document.
            document_id: Generated UUID string for the document.
            filename: Original filename.
            pages_data: Optional page-by-page metadata from PDF parsing.

        Returns:
            List of metadata-tagged DocumentChunk objects.
        """
        if not text or not text.strip():
            logger.warning(f"Empty text passed for document_id={document_id}")
            return []

        raw_chunks = self.splitter.split_text(text)
        document_chunks: list[DocumentChunk] = []

        for idx, chunk_text in enumerate(raw_chunks):
            meta: dict[str, Any] = {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": idx,
            }

            # Optional page mapping if pages_data available
            if pages_data:
                matching_pages = [
                    page["page_number"]
                    for page in pages_data
                    if page.get("text") and page["text"] in chunk_text or chunk_text[:50] in page.get("text", "")
                ]
                if matching_pages:
                    meta["page_number"] = matching_pages[0]

            document_chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    filename=filename,
                    chunk_index=idx,
                    text=chunk_text,
                    metadata=meta
                )
            )

        logger.info(
            f"Chunked document_id={document_id} ({filename}) into {len(document_chunks)} chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return document_chunks
