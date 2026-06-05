"""
==============================================================================
FAISS Vector Store — Index management for semantic search
==============================================================================

WHY FAISS:
    FAISS (Facebook AI Similarity Search) is the gold standard for dense
    vector search. For our use case it's ideal because:

    - Zero infrastructure: No server process, no database to manage.
      It's just a Python library that reads/writes index files to disk.
    - Blazing fast: Even the exact-search (Flat) index handles 100k vectors
      in milliseconds on CPU.
    - Simple persistence: save_local() / load_local() serialize the entire
      index to two files (index.faiss + index.pkl).
    - Native LangChain integration: FAISS is a first-class citizen in
      LangChain's vector store ecosystem.

    Alternatives considered:
        - ChromaDB: Needs a server process, heavier dependency
        - Pinecone/Weaviate: Cloud-based, paid, violates our "fully free" rule
        - Qdrant: Great but overkill for single-user local usage

DESIGN DECISIONS:
    - We use FAISS.from_documents() for initial index creation — it handles
      embedding generation and index building in one call.
    - Index persistence uses LangChain's save_local/load_local for simplicity.
    - We support INCREMENTAL adds: new papers are appended to an existing
      index without rebuilding from scratch.
    - The index path is configurable via settings.py for testing flexibility.

TRADEOFFS:
    - FAISS Flat index (exact search) is used by default. For >100k vectors,
      switch to IVFFlat (approximate search) for 10x speedup at ~5% recall cost.
    - No built-in metadata filtering — we handle this in the retrieval layer.
      FAISS is a pure vector search engine; filtering by source/page happens
      post-retrieval in retriever.py.
==============================================================================
"""

import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import FAISS_INDEX_DIR, ensure_directories
from embeddings.embedder import get_embedding_model

logger = logging.getLogger(__name__)

# Default index name for persistence (LangChain uses this as a filename prefix)
DEFAULT_INDEX_NAME = "paper_tutor_index"


def create_index(
    documents: list[Document],
    embedding_model: HuggingFaceEmbeddings | None = None,
) -> FAISS:
    """
    Create a new FAISS index from a list of Document chunks.

    This is the main entry point after the ingestion pipeline produces chunks.
    It embeds all documents and builds a flat (exact) FAISS index.

    Args:
        documents: List of chunked Document objects with page_content and metadata.
        embedding_model: Optional pre-loaded embedding model. Loads default if None.

    Returns:
        FAISS vector store instance (in-memory, not yet persisted).

    Raises:
        ValueError: If documents list is empty.

    Example:
        chunks = ingest_pdf("paper.pdf")["chunks"]
        index = create_index(chunks)
        index.save_local("data/faiss_index")
    """
    if not documents:
        raise ValueError("Cannot create index from empty document list")

    embedder = embedding_model or get_embedding_model()

    logger.info(f"Creating FAISS index from {len(documents)} documents...")

    # FAISS.from_documents handles:
    #   1. Embedding all document texts via the embedder
    #   2. Building the FAISS index (Flat by default — exact search)
    #   3. Storing document metadata alongside vectors
    index = FAISS.from_documents(
        documents=documents,
        embedding=embedder,
    )

    logger.info(
        f"FAISS index created: {index.index.ntotal} vectors, "
        f"dimension={index.index.d}"
    )

    return index


def save_index(
    index: FAISS,
    path: str | Path | None = None,
    index_name: str = DEFAULT_INDEX_NAME,
) -> Path:
    """
    Persist a FAISS index to disk.

    Creates two files:
        - {index_name}.faiss — the raw FAISS index (vectors + search structure)
        - {index_name}.pkl — pickled docstore (document texts + metadata)

    Args:
        index: FAISS vector store instance to save.
        path: Directory to save to. Default from settings.
        index_name: Filename prefix for the index files.

    Returns:
        Path to the save directory.
    """
    save_dir = Path(path) if path else FAISS_INDEX_DIR
    ensure_directories()
    save_dir.mkdir(parents=True, exist_ok=True)

    index.save_local(
        folder_path=str(save_dir),
        index_name=index_name,
    )

    logger.info(f"FAISS index saved to {save_dir}/{index_name}.*")

    return save_dir


def load_index(
    path: str | Path | None = None,
    index_name: str = DEFAULT_INDEX_NAME,
    embedding_model: HuggingFaceEmbeddings | None = None,
) -> FAISS:
    """
    Load a previously saved FAISS index from disk.

    Args:
        path: Directory containing the index files. Default from settings.
        index_name: Filename prefix used when saving.
        embedding_model: Optional pre-loaded embedding model.

    Returns:
        FAISS vector store instance loaded from disk.

    Raises:
        FileNotFoundError: If the index directory or files don't exist.
    """
    load_dir = Path(path) if path else FAISS_INDEX_DIR
    embedder = embedding_model or get_embedding_model()

    # Check that the index files exist
    faiss_file = load_dir / f"{index_name}.faiss"
    pkl_file = load_dir / f"{index_name}.pkl"

    if not faiss_file.exists() or not pkl_file.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {load_dir}/{index_name}.*. "
            f"Expected files: {faiss_file.name}, {pkl_file.name}. "
            f"Have you ingested any papers yet?"
        )

    index = FAISS.load_local(
        folder_path=str(load_dir),
        embeddings=embedder,
        index_name=index_name,
        allow_dangerous_deserialization=True,  # Required for pickle-based docstore
    )

    logger.info(
        f"FAISS index loaded from {load_dir}: "
        f"{index.index.ntotal} vectors, dimension={index.index.d}"
    )

    return index


def add_documents(
    index: FAISS,
    documents: list[Document],
) -> FAISS:
    """
    Add new documents to an existing FAISS index (incremental update).

    WHY INCREMENTAL ADDS:
        When a user uploads a second paper, we don't want to re-embed and
        re-index the first paper. We simply append the new vectors to the
        existing index. This is O(n) in the new documents only.

    Args:
        index: Existing FAISS vector store to extend.
        documents: New documents to add.

    Returns:
        The updated FAISS index (same object, modified in-place).
    """
    if not documents:
        logger.warning("No documents to add")
        return index

    before_count = index.index.ntotal

    index.add_documents(documents)

    after_count = index.index.ntotal
    logger.info(
        f"Added {after_count - before_count} vectors to index "
        f"(total: {after_count})"
    )

    return index


def delete_index(path: str | Path | None = None) -> bool:
    """
    Delete a persisted FAISS index from disk.

    Useful when the user wants to start fresh or remove a paper.

    Args:
        path: Directory containing the index files. Default from settings.

    Returns:
        True if index was deleted, False if it didn't exist.
    """
    delete_dir = Path(path) if path else FAISS_INDEX_DIR

    if not delete_dir.exists():
        logger.info(f"No index to delete at {delete_dir}")
        return False

    import shutil
    shutil.rmtree(delete_dir)
    delete_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Deleted FAISS index at {delete_dir}")
    return True


def similarity_search(
    index: FAISS,
    query: str,
    k: int = 4,
) -> list[tuple[Document, float]]:
    """
    Search the FAISS index for the most similar documents to a query.

    Returns documents WITH their similarity scores, which is useful for:
        - Score thresholding (filtering low-relevance results)
        - Displaying confidence to the user
        - Debugging retrieval quality

    Args:
        index: FAISS vector store to search.
        query: User's question or search query.
        k: Number of results to return.

    Returns:
        List of (Document, score) tuples, sorted by relevance (lowest score = most similar).
        Note: FAISS uses L2 distance by default, so LOWER scores = MORE similar.
        With normalized embeddings, score ≈ 2 * (1 - cosine_similarity).
    """
    results = index.similarity_search_with_score(query, k=k)

    logger.info(
        f"Search for '{query[:50]}...' returned {len(results)} results"
        if len(query) > 50
        else f"Search for '{query}' returned {len(results)} results"
    )

    return results


def get_index_stats(index: FAISS) -> dict:
    """
    Get statistics about a FAISS index.

    Useful for the Streamlit UI sidebar to show index health.

    Returns:
        Dict with: total_vectors, dimension, index_type.
    """
    return {
        "total_vectors": index.index.ntotal,
        "dimension": index.index.d,
        "index_type": type(index.index).__name__,
    }


def index_exists(path: str | Path | None = None, index_name: str = DEFAULT_INDEX_NAME) -> bool:
    """
    Check if a persisted FAISS index exists on disk.

    Args:
        path: Directory to check. Default from settings.
        index_name: Filename prefix to look for.

    Returns:
        True if both .faiss and .pkl files exist.
    """
    check_dir = Path(path) if path else FAISS_INDEX_DIR
    faiss_file = check_dir / f"{index_name}.faiss"
    pkl_file = check_dir / f"{index_name}.pkl"

    return faiss_file.exists() and pkl_file.exists()
