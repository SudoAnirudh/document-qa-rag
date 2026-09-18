class RAGException(Exception):
    """Base exception for Document Q&A RAG System."""
    pass


class InvalidFileError(RAGException):
    """Raised when an uploaded document file is invalid, corrupt, or unsupported."""
    pass


class ExternalServiceError(RAGException):
    """Raised when an external API (OpenAI embedding/generation) fails or times out."""
    pass
