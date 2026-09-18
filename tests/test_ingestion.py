import io
import pytest
from pathlib import Path
from pypdf import PdfWriter
from app.core.exceptions import InvalidFileError
from app.services.ingestion import IngestionService
from app.core.config import settings


def create_sample_pdf_bytes(text: str = "Hello World PDF Content") -> bytes:
    """Helper to generate valid PDF file bytes in memory using pypdf."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    # Write empty page or add annotation text
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_validate_file_extensions() -> None:
    """Verify validation of file extensions."""
    assert IngestionService.validate_file("doc.txt", b"some text") == ".txt"
    assert IngestionService.validate_file("doc.PDF", b"some bytes") == ".pdf"

    with pytest.raises(InvalidFileError, match="Unsupported file extension"):
        IngestionService.validate_file("doc.docx", b"some bytes")

    with pytest.raises(InvalidFileError, match="empty"):
        IngestionService.validate_file("doc.txt", b"")


def test_parse_valid_txt() -> None:
    """Verify parsing valid UTF-8 TXT files."""
    content = "This is a test document.\nLine 2 content."
    file_bytes = content.encode("utf-8")
    parsed = IngestionService.parse_txt(file_bytes)
    assert parsed == content


def test_parse_invalid_utf8_txt() -> None:
    """Verify invalid UTF-8 bytes raise InvalidFileError."""
    bad_bytes = b"\x80\x81\x82"
    with pytest.raises(InvalidFileError, match="valid UTF-8"):
        IngestionService.parse_txt(bad_bytes)


def test_parse_empty_txt() -> None:
    """Verify empty or whitespace-only TXT raises InvalidFileError."""
    with pytest.raises(InvalidFileError, match="no text"):
        IngestionService.parse_txt(b"   \n  ")


def test_parse_corrupt_pdf() -> None:
    """Verify corrupt PDF bytes raise InvalidFileError."""
    corrupt_bytes = b"%PDF-1.4 corrupt content here"
    with pytest.raises(InvalidFileError, match="corrupt or cannot be parsed"):
        IngestionService.parse_pdf(corrupt_bytes)


def test_process_document_txt_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify end-to-end process_document for a valid TXT file."""
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))

    filename = "sample.txt"
    file_bytes = b"Hello from ingestion test."

    doc_id, clean_name, text, pages = IngestionService.process_document(filename, file_bytes)

    assert len(doc_id) == 32  # UUID hex length
    assert clean_name == "sample.txt"
    assert text == "Hello from ingestion test."
    assert len(pages) == 1
    assert pages[0]["page_number"] == 1

    saved_file = tmp_path / f"{doc_id}_{clean_name}"
    assert saved_file.exists()
    assert saved_file.read_bytes() == file_bytes
