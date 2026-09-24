from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment or .env file."""

    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str | None = None
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_GENERATION_MODEL: str = "gpt-4o-mini"

    CHROMA_PATH: str = "./chroma_db"
    DATA_PATH: str = "./data"

    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 50
    DEFAULT_TOP_K: int = 5
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def ensure_directories(self) -> None:
        """Ensure necessary data and database storage directories exist."""
        Path(self.CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.DATA_PATH).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
