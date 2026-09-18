from app.models.schemas import SourceChunk
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.core.logging import logger


class RetrievalService:
    """Service wrapping semantic vector retrieval with optional document filtering."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store_service: VectorStoreService | None = None
    ) -> None:
        self.embedding_service = embedding_service if embedding_service is not None else EmbeddingService()
        self.vector_store_service = vector_store_service if vector_store_service is not None else VectorStoreService()

    def retrieve(
        self,
        question: str,
        document_id: str | None = None,
        top_k: int = 5
    ) -> list[SourceChunk]:
        """Embed question and query persistent vector store for top_k most relevant chunks.

        Args:
            question: Search question string.
            document_id: Optional document ID to filter search space.
            top_k: Max number of top chunks to return (1..20).

        Returns:
            List of SourceChunk objects containing document_id, chunk_index, text, and similarity score.
        """
        if not question or not question.strip():
            logger.warning("Empty question passed to RetrievalService.")
            return []

        # Step 1: Embed question
        query_vector = self.embedding_service.embed_query(question)

        # Step 2: Query Chroma persistent vector store
        raw_results = self.vector_store_service.query_similarity(
            query_embedding=query_vector,
            top_k=top_k,
            document_id=document_id
        )

        sources: list[SourceChunk] = []
        for res in raw_results:
            sources.append(
                SourceChunk(
                    document_id=res.get("document_id", ""),
                    chunk_index=res.get("chunk_index", 0),
                    text=res.get("text", ""),
                    score=res.get("score", 0.0)
                )
            )

        logger.info(
            f"Retrieval complete for question='{question[:35]}...': "
            f"retrieved {len(sources)} chunks (doc_id_filter={document_id}, top_k={top_k})"
        )
        return sources
