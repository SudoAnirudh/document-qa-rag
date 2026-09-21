from typing import Any
import chromadb
from app.core.config import settings
from app.core.logging import logger
from app.services.chunking import DocumentChunk


class VectorStoreService:
    """Service wrapping ChromaDB persistent storage and document vector retrieval."""

    COLLECTION_NAME = "document_qa_collection"

    def __init__(self, chroma_path: str | None = None) -> None:
        self.chroma_path = chroma_path if chroma_path is not None else settings.CHROMA_PATH
        self._client: chromadb.PersistentClient | None = None

    @property
    def client(self) -> chromadb.PersistentClient:
        """Lazy initialization of Chroma PersistentClient."""
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.chroma_path)
        return self._client

    def get_collection(self) -> Any:
        """Get or create the persistent collection using cosine distance metric."""
        return self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> int:
        """Store document chunks and their embedding vectors in Chroma persistent store.

        Args:
            chunks: List of DocumentChunk objects.
            embeddings: Parallel list of embedding vectors.

        Returns:
            Number of chunks added.
        """
        if not chunks or not embeddings or len(chunks) != len(embeddings):
            logger.warning("Empty or mismatched chunks/embeddings provided to VectorStoreService.")
            return 0

        ids = [f"{chunk.document_id}:{chunk.chunk_index}" for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        collection = self.get_collection()
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        logger.info(f"Added {len(chunks)} chunks to persistent Chroma collection '{self.COLLECTION_NAME}'")
        return len(chunks)

    def query_similarity(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Query persistent collection for top_k most similar chunks.

        Args:
            query_embedding: Search query embedding vector.
            top_k: Number of results to retrieve (1..20).
            document_id: Optional document_id filter.

        Returns:
            List of result dicts sorted by similarity score descending:
            [{"document_id": ..., "chunk_index": ..., "text": ..., "score": ...}]
        """
        if query_embedding is None or len(query_embedding) == 0:
            return []

        collection = self.get_collection()
        total_count = collection.count()

        if total_count == 0:
            logger.info("Chroma collection is empty.")
            return []

        n_results = min(top_k, total_count)
        where_filter = {"document_id": document_id} if document_id else None

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        retrieved: list[dict[str, Any]] = []

        if not results or not results.get("ids") or not results["ids"][0]:
            return retrieved

        ids_list = results["ids"][0]
        documents_list = results["documents"][0] if results.get("documents") else []
        metadatas_list = results["metadatas"][0] if results.get("metadatas") else []
        distances_list = results["distances"][0] if results.get("distances") else []

        for idx in range(len(ids_list)):
            meta = metadatas_list[idx] if idx < len(metadatas_list) else {}
            dist_val = float(distances_list[idx]) if idx < len(distances_list) else 1.0
            doc_text = documents_list[idx] if idx < len(documents_list) else ""

            # In Chroma cosine space, similarity score = 1.0 - cosine_distance
            similarity = round(max(0.0, min(1.0, 1.0 - dist_val)), 4)

            retrieved.append({
                "document_id": meta.get("document_id", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "text": doc_text,
                "score": similarity,
                "raw_distance": dist_val
            })

        retrieved.sort(key=lambda x: x["score"], reverse=True)
        return retrieved
