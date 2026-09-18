from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from app.api.routes import router
from app.core.exceptions import RAGException, InvalidFileError, ExternalServiceError
from app.core.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Document Q&A RAG System starting up...")
    yield
    logger.info("Document Q&A RAG System shutting down...")


app = FastAPI(
    title="Document Q&A RAG System",
    description="A small, production-minded RAG service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(InvalidFileError)
async def invalid_file_exception_handler(request: Request, exc: InvalidFileError) -> JSONResponse:
    logger.warning(f"Invalid file error on path {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )


@app.exception_handler(ExternalServiceError)
async def external_service_exception_handler(request: Request, exc: ExternalServiceError) -> JSONResponse:
    logger.error(f"External service failure on path {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": str(exc)}
    )


@app.exception_handler(RAGException)
async def general_rag_exception_handler(request: Request, exc: RAGException) -> JSONResponse:
    logger.error(f"RAG system exception on path {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal RAG system error"}
    )


app.include_router(router)


