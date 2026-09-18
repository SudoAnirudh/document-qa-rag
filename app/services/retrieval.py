import time
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

    def retrieve_with_metrics(
        self,
        question: str,
        document_id: str | None = None,
        top_k: int = 5
    ) -> tuple[list[SourceChunk], float, float]:
        """Embed question and query vector store for top_k chunks, returning (sources, embedding_ms, retrieval_ms)."""
        if not question or not question.strip():
            logger.warning("Empty question passed to RetrievalService.")
            return [], 0.0, 0.0

        t0 = time.perf_counter()
        query_vector = self.embedding_service.embed_query(question)
        t_embed = time.perf_counter()
        embedding_ms = (t_embed - t0) * 1000.0

        raw_results = self.vector_store_service.query_similarity(
            query_embedding=query_vector,
            top_k=top_k,
            document_id=document_id
        )
        t_ret = time.perf_counter()
        retrieval_ms = (t_ret - t_embed) * 1000.0

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
        return sources, embedding_ms, retrieval_ms

    def retrieve(
        self,
        question: str,
        document_id: str | None = None,
        top_k: int = 5
    ) -> list[SourceChunk]:
        """Embed question and query persistent vector store for top_k most relevant chunks."""
        sources, _, _ = self.retrieve_with_metrics(question, document_id, top_k)
        return sources

