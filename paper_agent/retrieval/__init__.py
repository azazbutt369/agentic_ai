# Retrieval package — semantic search and context assembly
#
# Public API:
#   retrieve(index, query) → list[RetrievalResult]
#   format_context(results) → str
#   get_sources_summary(results) → list[dict]
#
from retrieval.retriever import (
    retrieve,
    format_context,
    get_sources_summary,
    get_unique_sources,
    RetrievalResult,
    normalize_l2_score,
)

__all__ = [
    "retrieve",
    "format_context",
    "get_sources_summary",
    "get_unique_sources",
    "RetrievalResult",
    "normalize_l2_score",
]
