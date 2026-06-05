"""
==============================================================================
Ingestion Pipeline — End-to-end PDF → Chunks orchestrator
==============================================================================

WHY A PIPELINE ORCHESTRATOR:
    The three ingestion steps (load → clean → chunk) are always used together
    in sequence. This module provides a single entry point so callers
    (Streamlit UI, CLI scripts) don't need to know the internal wiring.

    It also handles:
        - Saving uploaded PDFs to the data directory
        - Logging the full pipeline execution
        - Returning both chunks and metadata for the UI
==============================================================================
"""

import logging
import shutil
from pathlib import Path

from langchain_core.documents import Document

from config.settings import PAPERS_DIR, ensure_directories
from ingestion.pdf_loader import load_pdf, get_pdf_metadata
from ingestion.text_cleaner import clean_documents
from ingestion.chunker import chunk_documents, get_chunk_stats

logger = logging.getLogger(__name__)


def save_uploaded_pdf(file_name: str, file_bytes: bytes) -> Path:
    """
    Save an uploaded PDF (from Streamlit file_uploader) to the papers directory.

    Args:
        file_name: Original filename (e.g., "attention.pdf").
        file_bytes: Raw bytes of the uploaded file.

    Returns:
        Path to the saved file.
    """
    ensure_directories()
    dest = PAPERS_DIR / file_name

    dest.write_bytes(file_bytes)
    logger.info(f"Saved uploaded PDF to {dest}")

    return dest


def ingest_pdf(
    file_path: str | Path,
    strip_references: bool = False,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict:
    """
    Run the full ingestion pipeline: Load → Clean → Chunk.

    Args:
        file_path: Path to the PDF file.
        strip_references: Whether to remove the references section.
        chunk_size: Override chunk size in tokens.
        chunk_overlap: Override chunk overlap in tokens.

    Returns:
        Dict with:
            - chunks: List[Document] — ready for embedding
            - metadata: dict — PDF-level metadata (title, author, pages)
            - stats: dict — chunk statistics (count, avg size, etc.)
    """
    path = Path(file_path)
    logger.info(f"=== Ingestion Pipeline: {path.name} ===")

    # Step 1: Load PDF → page-level Documents
    logger.info("Step 1/3: Loading PDF...")
    pages = load_pdf(path)

    # Step 2: Clean text
    logger.info("Step 2/3: Cleaning text...")
    cleaned_pages = clean_documents(pages, strip_references=strip_references)

    # Step 3: Chunk into retrieval-ready pieces
    logger.info("Step 3/3: Chunking...")
    chunks = chunk_documents(
        cleaned_pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Gather metadata and stats
    metadata = get_pdf_metadata(path)
    stats = get_chunk_stats(chunks)

    logger.info(
        f"=== Pipeline complete: {stats['total_chunks']} chunks from "
        f"{metadata['pages']} pages ==="
    )

    return {
        "chunks": chunks,
        "metadata": metadata,
        "stats": stats,
    }
