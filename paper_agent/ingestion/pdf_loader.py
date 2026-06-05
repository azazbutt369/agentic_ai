"""
==============================================================================
PDF Loader — Extract raw text from research paper PDFs
==============================================================================

WHY PyMuPDF (fitz)?
    - Fastest Python PDF parser (~10x faster than PyPDF2)
    - Best text extraction quality for academic papers
    - Handles multi-column layouts better than alternatives
    - Preserves reading order in most cases
    - Extracts metadata (title, author, etc.) from PDF properties
    - Lightweight — no Java/external dependencies (unlike Tika)

DESIGN DECISIONS:
    - We extract text PAGE-BY-PAGE and attach page numbers as metadata.
      This is critical for citation-aware responses ("Source: paper.pdf, Page 5").
    - We return LangChain `Document` objects so they plug directly into
      the downstream chunking and embedding pipeline.
    - We skip pages with very little text (likely figures/diagrams only)
      to avoid polluting the index with noise.

TRADEOFFS:
    - PyMuPDF extracts TEXT only — tables and figures are lost.
      For table extraction, you'd need Camelot or Tabula (future enhancement).
    - Multi-column papers sometimes interleave columns. PyMuPDF handles
      this well in most cases, but very complex layouts may need post-processing.
==============================================================================
"""

import logging
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Minimum character count for a page to be considered "real" content.
# Pages with fewer chars are usually blank, title-only, or figure-only pages.
MIN_PAGE_CHARS = 50


def load_pdf(file_path: str | Path) -> list[Document]:
    """
    Load a PDF file and extract text as a list of LangChain Documents.

    Each Document represents one page and carries metadata:
        - source: filename (not full path, for cleaner citations)
        - page: 1-indexed page number
        - total_pages: total pages in the PDF

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of Document objects, one per content-bearing page.

    Raises:
        FileNotFoundError: If the PDF file doesn't exist.
        ValueError: If the file is not a PDF or contains no extractable text.
    """
    path = Path(file_path)

    # --- Validation ---
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    # --- Extract text ---
    documents: list[Document] = []

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        raise ValueError(f"Failed to open PDF '{path.name}': {e}")

    total_pages = len(doc)
    logger.info(f"Opened '{path.name}' — {total_pages} pages")

    for page_num in range(total_pages):
        page = doc[page_num]

        # Extract text with preserved layout whitespace
        text = page.get_text("text")

        # Skip near-empty pages (figures, blank pages, etc.)
        if len(text.strip()) < MIN_PAGE_CHARS:
            logger.debug(
                f"Skipping page {page_num + 1} — only {len(text.strip())} chars"
            )
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,          # e.g., "attention_is_all_you_need.pdf"
                    "page": page_num + 1,          # 1-indexed for human readability
                    "total_pages": total_pages,
                },
            )
        )

    doc.close()

    if not documents:
        raise ValueError(
            f"No extractable text found in '{path.name}'. "
            "The PDF might be image-based (scanned). "
            "Consider using OCR (e.g., Tesseract) as a preprocessing step."
        )

    logger.info(
        f"Extracted {len(documents)} content pages from '{path.name}' "
        f"(skipped {total_pages - len(documents)} empty/figure pages)"
    )

    return documents


def get_pdf_metadata(file_path: str | Path) -> dict:
    """
    Extract PDF-level metadata (title, author, creation date, etc.).

    Useful for display in the Streamlit UI sidebar.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Dict with keys: title, author, subject, creator, pages.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(str(path))
    meta = doc.metadata

    result = {
        "title": meta.get("title", "").strip() or path.stem,
        "author": meta.get("author", "").strip() or "Unknown",
        "subject": meta.get("subject", "").strip() or "",
        "creator": meta.get("creator", "").strip() or "",
        "pages": len(doc),
    }

    doc.close()
    return result
