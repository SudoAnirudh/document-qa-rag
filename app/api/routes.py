import time
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from app.models.schemas import UploadResponse, QueryRequest, QueryResponse, SourceChunk
from app.core.config import settings
from app.core.exceptions import InvalidFileError, ExternalServiceError
from app.core.logging import logger
from app.services.ingestion import IngestionService
from app.services.chunking import ChunkingService
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.retrieval import RetrievalService
from app.services.generation import GenerationService


router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness check endpoint returning HTTP 200 with status ok."""
    return {"status": "ok"}


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_200_OK)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Accept PDF/TXT document upload, chunk text, embed chunks, and persist to vector store."""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file must have a filename.")

    try:
        file_bytes = await file.read()

        # Step 1: Ingestion & parsing
        document_id, clean_filename, full_text, pages_data = IngestionService.process_document(
            file.filename,
            file_bytes
        )

        # Step 2: Configurable Chunking
        chunker = ChunkingService()
        chunks = chunker.chunk_document(
            text=full_text,
            document_id=document_id,
            filename=clean_filename,
            pages_data=pages_data
        )

        # Step 3: Embeddings
        embedder = EmbeddingService()
        embeddings = embedder.embed_documents([chunk.text for chunk in chunks])

        # Step 4: Persistent Vector Storage
        vector_store = VectorStoreService()
        vector_store.add_chunks(chunks, embeddings)

        logger.info(
            f"Successfully processed upload: document_id={document_id}, "
            f"filename={clean_filename}, chunk_count={len(chunks)}"
        )

        return UploadResponse(
            document_id=document_id,
            filename=clean_filename,
            chunk_count=len(chunks)
        )

    except InvalidFileError as e:
        logger.warning(f"Invalid document upload attempt ({file.filename}): {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ExternalServiceError as e:
        logger.error(f"External service failure during upload processing: {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing document upload ({file.filename}): {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during document upload"
        )


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_document(request: QueryRequest) -> QueryResponse:
    """Execute semantic retrieval, evaluate grounding threshold, and generate grounded answer with latency instrumentation."""
    t_start = time.perf_counter()
    embedding_ms = 0.0
    retrieval_ms = 0.0
    generation_ms = 0.0

    try:
        retriever = RetrievalService()
        t_ret_start = time.perf_counter()
        sources = retriever.retrieve(
            question=request.question,
            document_id=request.document_id,
            top_k=request.top_k
        )
        t_ret_end = time.perf_counter()
        retrieval_ms = (t_ret_end - t_ret_start) * 1000.0



        # Step 3: Grounding Threshold Evaluation
        max_score = max((s.score for s in sources), default=0.0)
        threshold = settings.DEFAULT_SIMILARITY_THRESHOLD

        if not sources or max_score < threshold:
            t_end = time.perf_counter()
            total_ms = (t_end - t_start) * 1000.0

            logger.info(
                f"query_metric embedding_ms={embedding_ms:.2f} retrieval_ms={retrieval_ms:.2f} "
                f"generation_ms={generation_ms:.2f} total_ms={total_ms:.2f} grounding_rejected=true"
            )
            return QueryResponse(
                answer="not found in the provided documents",
                sources=[],
                grounded=False
            )

        # Step 4: Grounded Answer Generation
        generator = GenerationService()
        t_gen_start = time.perf_counter()
        answer = generator.generate_answer(
            question=request.question,
            context_chunks=sources
        )
        t_gen_end = time.perf_counter()
        generation_ms = (t_gen_end - t_gen_start) * 1000.0

        total_ms = (t_gen_end - t_start) * 1000.0

        logger.info(
            f"query_metric embedding_ms={embedding_ms:.2f} retrieval_ms={retrieval_ms:.2f} "
            f"generation_ms={generation_ms:.2f} total_ms={total_ms:.2f} grounded=true"
        )

        return QueryResponse(
            answer=answer,
            sources=sources,
            grounded=True
        )

    except ExternalServiceError as e:
        logger.error(f"External API failure during query processing: {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing query request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during query processing"
        )


def retriever_result_to_source_chunk(res: dict) -> SourceChunk:
    """Helper mapping vector store dict result to SourceChunk model."""
    return SourceChunk(
        document_id=res.get("document_id", ""),
        chunk_index=res.get("chunk_index", 0),
        text=res.get("text", ""),
        score=res.get("score", 0.0)
    )




