import io
import uuid
from pathlib import Path
from pypdf import PdfReader
from app.core.config import settings
from app.core.exceptions import InvalidFileError
from app.core.logging import logger


class IngestionService:
    """Service responsible for file validation, parsing PDF/TXT documents, and local file storage."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    @staticmethod
    def generate_document_id() -> str:
        """Generate a unique hex UUID string for the uploaded document."""
        return uuid.uuid4().hex

    @classmethod
    def validate_file(cls, filename: str, file_bytes: bytes) -> str:
        """Validate filename extension and byte length.

        Returns lowercase extension (e.g. '.pdf', '.txt').
        Raises InvalidFileError if invalid.
        """
        if not filename or "." not in filename:
            raise InvalidFileError("Filename must include an extension (.pdf or .txt).")

        ext = Path(filename).suffix.lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise InvalidFileError(f"Unsupported file extension '{ext}'. Only .pdf and .txt are supported.")

        if not file_bytes or len(file_bytes.strip()) == 0:
            raise InvalidFileError("Uploaded file is empty.")

        return ext

    @staticmethod
    def parse_txt(file_bytes: bytes) -> str:
        """Decode and validate TXT file content as UTF-8."""
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.warning(f"UTF-8 decoding failed for TXT file: {e}")
            raise InvalidFileError("TXT file must be valid UTF-8 encoded text.")

        cleaned_text = text.strip()
        if not cleaned_text:
            raise InvalidFileError("TXT file contains no text.")

        return cleaned_text

    @staticmethod
    def parse_pdf(file_bytes: bytes) -> tuple[str, list[dict[str, int | str]]]:
        """Extract text from PDF file bytes page by page.

        Returns tuple of (full_text, page_metadata_list).
        Raises InvalidFileError if PDF is corrupt or has no extractable text.
        """
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            if not reader.pages:
                raise InvalidFileError("PDF file contains no pages.")
        except Exception as e:
            logger.warning(f"pypdf failed to parse PDF: {e}")
            raise InvalidFileError("PDF file is corrupt or cannot be parsed.")

        pages_data: list[dict[str, int | str]] = []
        full_text_parts: list[str] = []

        for idx, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"Error extracting text from PDF page {idx + 1}: {e}")
                page_text = ""

            page_text_clean = page_text.strip()
            if page_text_clean:
                full_text_parts.append(page_text_clean)
                pages_data.append({
                    "page_number": idx + 1,
                    "text": page_text_clean
                })

        full_text = "\n\n".join(full_text_parts).strip()
        if not full_text:
            raise InvalidFileError("PDF file contains no extractable text.")

        return full_text, pages_data

    @classmethod
    def process_document(cls, filename: str, file_bytes: bytes) -> tuple[str, str, str, list[dict[str, int | str]]]:
        """Orchestrate file validation, ID generation, parsing, and raw file saving.

        Returns tuple of (document_id, cleaned_filename, full_text, page_metadata_list).
        """
        ext = cls.validate_file(filename, file_bytes)
        document_id = cls.generate_document_id()
        clean_filename = Path(filename).name

        if ext == ".txt":
            full_text = cls.parse_txt(file_bytes)
            pages_data = [{"page_number": 1, "text": full_text}]
        elif ext == ".pdf":
            full_text, pages_data = cls.parse_pdf(file_bytes)
        else:
            raise InvalidFileError(f"Unsupported extension: {ext}")

        cls.save_raw_file(file_bytes, document_id, clean_filename)
        return document_id, clean_filename, full_text, pages_data

    @staticmethod
    def save_raw_file(file_bytes: bytes, document_id: str, filename: str) -> Path:
        """Save raw uploaded bytes to DATA_PATH/{document_id}_{filename}."""
        destination = Path(settings.DATA_PATH) / f"{document_id}_{filename}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(file_bytes)
        logger.info(f"Saved raw document to {destination}")
        return destination

