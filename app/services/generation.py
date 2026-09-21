from openai import OpenAI, APIError, APITimeoutError, RateLimitError, APIConnectionError
from app.core.config import settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import logger
from app.models.schemas import SourceChunk


class GenerationService:
    """Service wrapping OpenAI gpt-4o-mini generation calls with a strict grounding prompt."""

    SYSTEM_PROMPT = (
        "You answer questions using only the provided document context.\n\n"
        "Do not use outside knowledge.\n\n"
        "Do not infer facts that are not supported by the context.\n\n"
        "If the answer is not supported by the provided context, respond exactly:\n\n"
        '"not found in the provided documents"'
    )

    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key if api_key is not None else settings.OPENAI_API_KEY
        self.model = model if model is not None else settings.OPENAI_GENERATION_MODEL
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

    def generate_answer(self, question: str, context_chunks: list[SourceChunk]) -> str:
        """Generate a grounded answer using exclusively the provided context chunks.

        Args:
            question: User question string.
            context_chunks: List of retrieved SourceChunk objects.

        Returns:
            Generated answer string or 'not found in the provided documents'.

        Raises:
            ExternalServiceError: If OpenAI API fails, times out, or rate limits.
        """
        if not context_chunks:
            return "not found in the provided documents"

        formatted_chunks = [
            f"[Source {idx + 1} | Document: {chunk.document_id} | Chunk: {chunk.chunk_index}]\n{chunk.text}"
            for idx, chunk in enumerate(context_chunks)
        ]
        context_block = "\n\n---\n\n".join(formatted_chunks)

        user_prompt = (
            f"DOCUMENT CONTEXT:\n{context_block}\n\n"
            f"---\n\n"
            f"USER QUESTION:\n{question}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )

            answer = response.choices[0].message.content or ""
            return answer.strip()

        except (APITimeoutError, RateLimitError, APIConnectionError, APIError) as e:
            logger.error(f"OpenAI generation API error ({type(e).__name__}): {e}")
            raise ExternalServiceError("Generation service temporarily unavailable") from e
        except Exception as e:
            logger.error(f"Unexpected error in GenerationService: {e}")
            raise ExternalServiceError("Generation service temporarily unavailable") from e
