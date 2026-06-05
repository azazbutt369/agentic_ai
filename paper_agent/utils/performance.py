"""
==============================================================================
Optimization Utilities — Performance profiling and tuning helpers
==============================================================================

This module provides tools for measuring and optimizing system performance.
Use these when tuning chunk sizes, retrieval parameters, or diagnosing
slow queries.

USAGE:
    from utils.performance import time_it, profile_ingestion, profile_query
==============================================================================
"""

import time
import logging
from pathlib import Path
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------


@contextmanager
def timer(label: str = "Operation"):
    """
    Context manager that measures and logs execution time.

    Usage:
        with timer("PDF ingestion"):
            result = ingest_pdf("paper.pdf")
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start

    if elapsed < 1:
        logger.info(f"⏱️ {label}: {elapsed * 1000:.1f}ms")
    else:
        logger.info(f"⏱️ {label}: {elapsed:.2f}s")


def time_it(func):
    """
    Decorator that logs function execution time.

    Usage:
        @time_it
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"⏱️ {func.__name__}: {elapsed:.3f}s")
        return result
    return wrapper


# ---------------------------------------------------------------------------
# Profiling functions
# ---------------------------------------------------------------------------


def profile_ingestion(pdf_path: str | Path) -> dict:
    """
    Profile the full ingestion pipeline with timing for each step.

    Returns:
        Dict with timings: load_ms, clean_ms, chunk_ms, total_ms, stats.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from ingestion.pdf_loader import load_pdf
    from ingestion.text_cleaner import clean_documents
    from ingestion.chunker import chunk_documents

    timings = {}

    # Step 1: Load
    start = time.perf_counter()
    pages = load_pdf(pdf_path)
    timings["load_ms"] = round((time.perf_counter() - start) * 1000, 1)

    # Step 2: Clean
    start = time.perf_counter()
    cleaned = clean_documents(pages)
    timings["clean_ms"] = round((time.perf_counter() - start) * 1000, 1)

    # Step 3: Chunk
    start = time.perf_counter()
    chunks = chunk_documents(cleaned)
    timings["chunk_ms"] = round((time.perf_counter() - start) * 1000, 1)

    timings["total_ms"] = timings["load_ms"] + timings["clean_ms"] + timings["chunk_ms"]
    timings["pages"] = len(pages)
    timings["chunks"] = len(chunks)
    timings["avg_chunk_chars"] = (
        sum(len(c.page_content) for c in chunks) // len(chunks) if chunks else 0
    )

    return timings


def profile_query(index, query: str, k: int = 4) -> dict:
    """
    Profile a single query through retrieval + generation.

    Args:
        index: FAISS vector store.
        query: Test query string.
        k: Number of results.

    Returns:
        Dict with timings: retrieve_ms, format_ms, total_ms, num_results.
    """
    from retrieval.retriever import retrieve, format_context

    timings = {}

    # Retrieval
    start = time.perf_counter()
    results = retrieve(index, query, k=k)
    timings["retrieve_ms"] = round((time.perf_counter() - start) * 1000, 1)

    # Formatting
    start = time.perf_counter()
    context = format_context(results)
    timings["format_ms"] = round((time.perf_counter() - start) * 1000, 1)

    timings["total_ms"] = timings["retrieve_ms"] + timings["format_ms"]
    timings["num_results"] = len(results)
    timings["context_chars"] = len(context)

    return timings


def estimate_memory_usage() -> dict:
    """
    Estimate current memory usage of key components.

    Returns:
        Dict with: python_mb, total_mb.
    """
    import psutil
    import os

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    return {
        "python_rss_mb": round(mem_info.rss / 1024 / 1024, 1),
        "python_vms_mb": round(mem_info.vms / 1024 / 1024, 1),
    }
