from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response returned upon successful document upload and indexing."""

    document_id: str
    filename: str
    chunk_count: int


class QueryRequest(BaseModel):
    """Request payload for semantic querying across uploaded documents."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User query string, must be between 1 and 2000 characters."
    )
    document_id: str | None = Field(
        default=None,
        description="Optional document ID to restrict search to a specific document."
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top context chunks to retrieve (between 1 and 20)."
    )


class SourceChunk(BaseModel):
    """Retrieved source chunk citation metadata."""

    document_id: str
    chunk_index: int
    text: str
    score: float


class QueryResponse(BaseModel):
    """Response containing answer, source chunk citations, and grounding status."""

    answer: str
    sources: list[SourceChunk]
    grounded: bool
