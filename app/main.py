from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from app.api.routes import router
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

app.include_router(router)

