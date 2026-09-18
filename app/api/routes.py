from fastapi import APIRouter, File, UploadFile, HTTPException, status
from app.models.schemas import UploadResponse
from app.core.exceptions import InvalidFileError, ExternalServiceError
from app.core.logging import logger
from app.services.ingestion import IngestionService
from app.services.chunking import ChunkingService
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStoreService

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

