"""
document_processor.py
---------------------
Document ingestion layer for DocuGuard+.

Supports PDF, DOCX, and TXT files.  Returns a structured result containing
the extracted text, cleaned text, and basic document metadata (word count,
sentence count, paragraph count).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List

import chardet
import docx
import fitz  # PyMuPDF

from utils.text_utils import (
    clean_text,
    count_words,
    split_paragraphs,
    split_sentences,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DocumentMetadata:
    """Basic statistics about an ingested document."""

    filename: str = ""
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    char_count: int = 0
    avg_sentence_length: float = 0.0


@dataclass
class DocumentResult:
    """Container returned by :func:`process_document`."""

    raw_text: str = ""
    cleaned_text: str = ""
    sentences: List[str] = field(default_factory=list)
    paragraphs: List[str] = field(default_factory=list)
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)


# ---------------------------------------------------------------------------
# Format-specific extractors
# ---------------------------------------------------------------------------

def _extract_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF byte-stream using PyMuPDF."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n\n".join(pages)


def _extract_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX byte-stream using python-docx."""
    document = docx.Document(io.BytesIO(file_bytes))
    paragraphs: list[str] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def _extract_txt(file_bytes: bytes) -> str:
    """Decode a plain-text byte-stream, auto-detecting encoding."""
    detection = chardet.detect(file_bytes)
    encoding = detection.get("encoding") or "utf-8"
    try:
        return file_bytes.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return file_bytes.decode("utf-8", errors="replace")


_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".txt": _extract_txt,
}

SUPPORTED_EXTENSIONS = set(_EXTRACTORS.keys())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_document(file_bytes: bytes, filename: str) -> DocumentResult:
    """Ingest a document and return structured text + metadata.

    Parameters
    ----------
    file_bytes:
        Raw bytes of the uploaded file.
    filename:
        Original filename (used to determine format via extension).

    Returns
    -------
    DocumentResult

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    ext = _get_extension(filename)
    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    raw_text = extractor(file_bytes)
    cleaned = clean_text(raw_text)
    sentences = split_sentences(cleaned)
    paragraphs = split_paragraphs(cleaned)
    wc = count_words(cleaned)

    meta = DocumentMetadata(
        filename=filename,
        word_count=wc,
        sentence_count=len(sentences),
        paragraph_count=len(paragraphs),
        char_count=len(cleaned),
        avg_sentence_length=wc / len(sentences) if sentences else 0.0,
    )

    return DocumentResult(
        raw_text=raw_text,
        cleaned_text=cleaned,
        sentences=sentences,
        paragraphs=paragraphs,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_extension(filename: str) -> str:
    """Return the lowercased extension including the dot."""
    idx = filename.rfind(".")
    if idx == -1:
        return ""
    return filename[idx:].lower()
