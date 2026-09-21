from openai import OpenAI, APIError, APITimeoutError, RateLimitError, APIConnectionError
from app.core.config import settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import logger


class EmbeddingService:
    """Service wrapping OpenAI text-embedding-3-small API calls with robust error handling."""

    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key if api_key is not None else settings.OPENAI_API_KEY
        self.model = model if model is not None else settings.OPENAI_EMBEDDING_MODEL
        self.base_url = base_url if base_url is not None else settings.OPENAI_BASE_URL
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            key_to_use = self.api_key or "missing_key"
            kwargs: dict[str, str] = {"api_key": key_to_use}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate vector embeddings for a list of document chunk texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding float vectors.

        Raises:
            ExternalServiceError: If OpenAI embedding API fails, times out, or rate limits.
        """
        if not texts:
            return []

        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            return [data.embedding for data in response.data]
        except (APITimeoutError, RateLimitError, APIConnectionError, APIError) as e:
            logger.warning(f"Embedding API error ({type(e).__name__}): {e}. Falling back to local ONNX embeddings.")
            try:
                from chromadb.utils import embedding_functions
                fn = embedding_functions.DefaultEmbeddingFunction()
                raw_embeds = fn(texts)
                return [[float(x) for x in emb] for emb in raw_embeds]
            except Exception as fallback_err:
                logger.error(f"Local embedding fallback failed: {fallback_err}")
                raise ExternalServiceError("Embedding service temporarily unavailable") from e
        except Exception as e:
            logger.error(f"Unexpected error in EmbeddingService: {e}")
            raise ExternalServiceError("Embedding service temporarily unavailable") from e

    def embed_query(self, text: str) -> list[float]:
        """Generate a single vector embedding for a search query string.

        Args:
            text: Query string.

        Returns:
            Embedding float vector.

        Raises:
            ExternalServiceError: If OpenAI embedding API fails, times out, or rate limits.
        """
        if not text or not text.strip():
            return []

        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []
