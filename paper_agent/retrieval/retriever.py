"""
==============================================================================
Retriever — Semantic search with score filtering and citation support
==============================================================================

WHY A RETRIEVER LAYER ON TOP OF FAISS:
    The FAISS store (Phase 3) provides raw vector search — "give me the k
    nearest vectors." But raw results need post-processing for quality RAG:

    1. SCORE THRESHOLDING: Not all "top-k" results are relevant. If a user
       asks about "reinforcement learning" but only Transformer papers are
       indexed, all results will be low-relevance noise. Thresholding filters
       these out so the LLM doesn't hallucinate from irrelevant context.

    2. MMR (Maximal Marginal Relevance): Pure similarity search often returns
       redundant chunks from the same paragraph. MMR re-ranks to balance
       relevance with diversity — critical for queries like "compare methods"
       that need chunks from DIFFERENT sections.

    3. SOURCE FILTERING: When multiple papers are indexed, users may want to
       ask about a specific paper. Source filtering restricts retrieval to
       chunks from a particular PDF.

    4. CITATION FORMATTING: Each retrieved chunk is annotated with
       "[Source: filename, Page X]" so the generation chain can produce
       citation-aware responses.

DESIGN DECISIONS:
    - This module wraps the FAISS store's search functions with additional
      filtering logic.
    - It returns a standardized `RetrievalResult` dataclass so downstream
      consumers (generation chain, UI) have a clean interface.
    - MMR is the default search mode because diversity matters more than
      raw similarity for RAG applications.
    - Score normalization: FAISS L2 distances are converted to a 0-1
      similarity score for human-interpretable thresholding.
==============================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config.settings import (
    TOP_K,
    SIMILARITY_THRESHOLD,
    USE_MMR,
    MMR_LAMBDA,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """
    A single retrieval result with metadata and relevance score.

    WHY a dataclass instead of raw tuples?
        - Type safety: no more guessing result[0] vs result[1]
        - Self-documenting: fields have clear names
        - Extensible: easy to add fields (e.g., reranker_score) later
    """
    document: Document           # The chunk's text and metadata
    similarity_score: float      # 0-1 normalized score (higher = more relevant)
    source: str = ""             # Filename (e.g., "attention.pdf")
    page: int = 0                # Page number in the source PDF
    chunk_index: int = 0         # Chunk index within the document
    citation: str = ""           # Formatted citation string

    def __post_init__(self):
        """Auto-populate source, page, and citation from document metadata."""
        meta = self.document.metadata
        self.source = meta.get("source", "unknown")
        self.page = meta.get("page", 0)
        self.chunk_index = meta.get("chunk_index", 0)
        self.citation = f"[Source: {self.source}, Page {self.page}]"


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------


def normalize_l2_score(l2_distance: float) -> float:
    """
    Convert FAISS L2 distance to a 0-1 similarity score.

    WHY NORMALIZATION:
        FAISS returns L2 (Euclidean) distances where LOWER = more similar.
        This is counter-intuitive for thresholding (where we want HIGHER = better).

        With L2-normalized embeddings (which we use), the relationship is:
            L2_distance = 2 * (1 - cosine_similarity)
            cosine_similarity = 1 - (L2_distance / 2)

        This gives us a 0-1 score where 1.0 = identical, 0.0 = orthogonal.

    Args:
        l2_distance: Raw L2 distance from FAISS.

    Returns:
        Normalized similarity score in [0, 1].
    """
    # Clamp to valid range (floating point can produce tiny negatives)
    similarity = max(0.0, 1.0 - (float(l2_distance) / 2.0))
    return round(similarity, 6)


# ---------------------------------------------------------------------------
# Core retrieval functions
# ---------------------------------------------------------------------------


def retrieve(
    index: FAISS,
    query: str,
    k: int | None = None,
    threshold: float | None = None,
    use_mmr: bool | None = None,
    mmr_lambda: float | None = None,
    source_filter: str | None = None,
) -> list[RetrievalResult]:
    """
    Retrieve the most relevant chunks for a query with filtering.

    This is the main retrieval function used by the generation chain.
    It combines search, score filtering, and source filtering into
    a single clean interface.

    Args:
        index: FAISS vector store to search.
        query: User's question or search text.
        k: Number of results to retrieve (default from settings).
        threshold: Minimum similarity score to include (default from settings).
        use_mmr: Whether to use MMR for diversity (default from settings).
        mmr_lambda: MMR diversity weight (default from settings).
        source_filter: If set, only return chunks from this source file.

    Returns:
        List of RetrievalResult objects, sorted by similarity (highest first).
    """
    num_results = k or TOP_K
    min_score = threshold if threshold is not None else SIMILARITY_THRESHOLD
    do_mmr = use_mmr if use_mmr is not None else USE_MMR
    lambda_mult = mmr_lambda if mmr_lambda is not None else MMR_LAMBDA

    if do_mmr:
        results = _search_mmr(index, query, num_results, lambda_mult)
    else:
        results = _search_similarity(index, query, num_results)

    # Apply score thresholding
    if min_score > 0:
        before_count = len(results)
        results = [r for r in results if r.similarity_score >= min_score]
        filtered = before_count - len(results)
        if filtered > 0:
            logger.debug(
                f"Score threshold {min_score} filtered out {filtered} results"
            )

    # Apply source filtering
    if source_filter:
        results = [r for r in results if r.source == source_filter]
        logger.debug(f"Source filter '{source_filter}' → {len(results)} results")

    # Sort by similarity (highest first)
    results.sort(key=lambda r: r.similarity_score, reverse=True)

    logger.info(
        f"Retrieved {len(results)} results for query: "
        f"'{query[:60]}{'...' if len(query) > 60 else ''}'"
    )

    return results


def _search_similarity(
    index: FAISS,
    query: str,
    k: int,
) -> list[RetrievalResult]:
    """
    Pure similarity search — returns the k nearest neighbors.

    Simple and fast, but may return redundant chunks from the
    same section of a paper.
    """
    raw_results = index.similarity_search_with_score(query, k=k)

    return [
        RetrievalResult(
            document=doc,
            similarity_score=normalize_l2_score(score),
        )
        for doc, score in raw_results
    ]


def _search_mmr(
    index: FAISS,
    query: str,
    k: int,
    lambda_mult: float,
) -> list[RetrievalResult]:
    """
    MMR (Maximal Marginal Relevance) search — balances relevance with diversity.

    WHY MMR:
        Pure similarity often returns 3-4 chunks from the SAME paragraph
        because they were created with overlapping content. MMR iteratively
        selects the next result that is:
            - Relevant to the query (high similarity)
            - Different from already-selected results (high diversity)

        The lambda_mult parameter controls the trade-off:
            - lambda=1.0 → pure similarity (no diversity)
            - lambda=0.5 → balanced (our default)
            - lambda=0.0 → pure diversity (ignores relevance)

    NOTE: LangChain's FAISS MMR doesn't return scores, so we assign
    approximate scores by doing a parallel similarity search.
    """
    # Get MMR-diverse documents
    mmr_docs = index.max_marginal_relevance_search(
        query=query,
        k=k,
        lambda_mult=lambda_mult,
        fetch_k=k * 3,  # Fetch 3x candidates for better diversity selection
    )

    # MMR doesn't return scores — do a parallel similarity search to get scores
    # for the same documents (by matching page_content)
    scored_results = index.similarity_search_with_score(query, k=k * 3)
    score_map = {}
    for doc, score in scored_results:
        # Use content hash as key to match documents
        key = doc.page_content[:100]
        if key not in score_map:
            score_map[key] = normalize_l2_score(score)

    results = []
    for doc in mmr_docs:
        key = doc.page_content[:100]
        score = score_map.get(key, 0.5)  # Default score if not found
        results.append(
            RetrievalResult(
                document=doc,
                similarity_score=score,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def format_context(results: list[RetrievalResult]) -> str:
    """
    Format retrieval results into a context string for the LLM prompt.

    Each chunk is wrapped with its citation so the LLM can reference
    specific sources in its response.

    Args:
        results: List of RetrievalResult objects.

    Returns:
        Formatted context string ready for prompt injection.

    Example output:
        --- [Source: attention.pdf, Page 3] ---
        The Transformer architecture uses self-attention...

        --- [Source: attention.pdf, Page 5] ---
        Multi-head attention allows the model to jointly attend...
    """
    if not results:
        return "No relevant context found."

    chunks = []
    for r in results:
        chunks.append(
            f"--- {r.citation} ---\n"
            f"{r.document.page_content}"
        )

    return "\n\n".join(chunks)


def get_sources_summary(results: list[RetrievalResult]) -> list[dict]:
    """
    Extract a summary of sources from retrieval results.

    Useful for displaying source chips/badges in the Streamlit UI.

    Args:
        results: List of RetrievalResult objects.

    Returns:
        List of dicts with: source, page, score, preview.
    """
    return [
        {
            "source": r.source,
            "page": r.page,
            "score": r.similarity_score,
            "preview": r.document.page_content[:150] + "...",
        }
        for r in results
    ]


def get_unique_sources(results: list[RetrievalResult]) -> list[str]:
    """
    Get unique source filenames from retrieval results.

    Args:
        results: List of RetrievalResult objects.

    Returns:
        Deduplicated list of source filenames.
    """
    seen = set()
    sources = []
    for r in results:
        if r.source not in seen:
            seen.add(r.source)
            sources.append(r.source)
    return sources
