"""
Document text extraction and chunking service.

Supports: PDF, DOCX, TXT, MD
Chunking: character-based with configurable size and overlap.
"""

import logging

from backend.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text(file_path: str, file_type: str) -> str:
    """Extract plain text from a document file.

    Args:
        file_path: Absolute path to the uploaded file.
        file_type: One of 'pdf', 'docx', 'txt', 'md'.

    Returns:
        Extracted text as a single string.

    Raises:
        ValueError: If the file type is unsupported or extraction fails.
    """
    file_type = file_type.lower()
    try:
        if file_type == "pdf":
            return _extract_pdf(file_path)
        elif file_type == "docx":
            return _extract_docx(file_path)
        elif file_type in ("txt", "md"):
            return _extract_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except ValueError:
        raise
    except Exception as e:
        logger.exception(f"Text extraction failed for {file_path}")
        raise ValueError(f"Failed to extract text: {e}")


def _extract_pdf(file_path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    text = "\n".join(pages)
    if not text.strip():
        raise ValueError("PDF contains no extractable text (may be scanned/image-only)")
    return text


def _extract_docx(file_path: str) -> str:
    from docx import Document as DocxDocument
    doc = DocxDocument(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)
    if not text.strip():
        raise ValueError("DOCX contains no extractable text")
    return text


def _extract_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if not text.strip():
        raise ValueError("File is empty")
    return text


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks.

    Uses character-based splitting. Each chunk is at most `chunk_size`
    characters, with `chunk_overlap` characters of overlap between
    consecutive chunks.

    Args:
        text: The full document text.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text.strip():
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        # If we're not at the end, try to break at a sentence/paragraph boundary
        if end < text_len:
            # Look for a good break point (newline, period+space) within the last 20% of the chunk
            search_start = start + int(chunk_size * 0.8)
            best_break = -1

            # Prefer paragraph breaks
            newline_pos = text.rfind("\n", search_start, end)
            if newline_pos != -1:
                best_break = newline_pos + 1

            # Fall back to sentence breaks
            if best_break == -1:
                for sep in (". ", "? ", "! "):
                    pos = text.rfind(sep, search_start, end)
                    if pos != -1 and pos > best_break:
                        best_break = pos + len(sep)

            if best_break > start:
                end = best_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward by (chunk_size - overlap), but at least 1 character
        step = max(end - start - chunk_overlap, 1)
        start += step

    return chunks
