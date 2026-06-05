# Vector store package — FAISS index management
#
# Public API:
#   create_index(docs) → FAISS
#   save_index(index) → Path
#   load_index() → FAISS
#   add_documents(index, docs) → FAISS
#   similarity_search(index, query) → list[(Document, float)]
#   index_exists() → bool
#
from vectorstore.faiss_store import (
    create_index,
    save_index,
    load_index,
    add_documents,
    delete_index,
    similarity_search,
    get_index_stats,
    index_exists,
)

__all__ = [
    "create_index",
    "save_index",
    "load_index",
    "add_documents",
    "delete_index",
    "similarity_search",
    "get_index_stats",
    "index_exists",
]
