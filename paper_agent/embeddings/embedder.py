"""
==============================================================================
Embedder — Convert text chunks into dense vector representations
==============================================================================

WHY EMBEDDINGS ARE THE HEART OF RAG:
    Embeddings are fixed-length numerical vectors that capture the MEANING
    of text. When a user asks "How does the Transformer handle long sequences?",
    we embed that question and find chunks whose embeddings are closest in
    vector space — those chunks are about the same CONCEPT, even if they use
    different words.

    This is what makes RAG work: semantic search instead of keyword matching.

WHY bge-small-en-v1.5:
    - 384-dimensional output — compact vectors = fast FAISS search
    - ~130 MB model size — loads in seconds, fits in RAM easily
    - Top-tier on MTEB benchmark for its size class (better than many
      models 3-5x larger)
    - Trained specifically for retrieval tasks (query-document matching)
    - English-focused — perfect for English-language ML papers

    Alternatives considered:
        - bge-base (768-dim): 2x larger vectors, marginal quality gain
        - all-MiniLM-L6-v2: Similar size but lower retrieval quality
        - bge-large (1024-dim): Overkill for our use case, 4x memory

DESIGN DECISIONS:
    - We wrap HuggingFaceEmbeddings from langchain_huggingface for
      seamless integration with the LangChain FAISS vector store.
    - Query prefixing: bge models benefit from prepending
      "Represent this sentence: " to queries (not documents). This is
      handled automatically by the model's encode configuration.
    - We use CPU by default but auto-detect CUDA if available.
    - The embedder is a SINGLETON — loading the model is expensive (~2s),
      so we cache it after first initialization.
==============================================================================
"""

import logging
from functools import lru_cache

import torch
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import EMBEDDING_MODEL, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    """
    Auto-detect the best available compute device.

    Priority: CUDA GPU → CPU
    MPS (Apple Silicon) is excluded because bge models have
    inconsistent MPS support in some torch versions.
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
    else:
        device = "cpu"
        logger.info("Using CPU for embeddings (no CUDA GPU detected)")

    return device


@lru_cache(maxsize=1)
def get_embedding_model(
    model_name: str | None = None,
    device: str | None = None,
) -> HuggingFaceEmbeddings:
    """
    Load and cache the embedding model (singleton pattern).

    The model is loaded once and reused across all calls. This avoids
    the ~2 second load time on every embedding operation.

    Args:
        model_name: HuggingFace model identifier. Default from settings.
        device: Compute device ("cpu" or "cuda"). Auto-detected if None.

    Returns:
        Configured HuggingFaceEmbeddings instance.

    WHY lru_cache:
        Loading a transformer model involves reading ~130 MB from disk,
        initializing weights, and moving them to the compute device.
        Caching ensures this happens only ONCE per session.
    """
    model = model_name or EMBEDDING_MODEL
    dev = device or _detect_device()

    logger.info(f"Loading embedding model: {model} on {dev}")

    # model_kwargs: passed to the underlying SentenceTransformer constructor
    # encode_kwargs: passed to model.encode() for every embedding call
    embeddings = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": dev},
        encode_kwargs={
            "normalize_embeddings": True,  # L2 normalize → cosine sim = dot product
            "batch_size": 64,              # Process 64 chunks at a time
        },
    )

    logger.info(f"Embedding model loaded: {model} ({EMBEDDING_DIMENSION}-dim)")

    return embeddings


def embed_texts(texts: list[str], model: HuggingFaceEmbeddings | None = None) -> list[list[float]]:
    """
    Embed a list of text strings into dense vectors.

    This is a convenience function for direct embedding without
    going through the FAISS vector store (useful for testing/debugging).

    Args:
        texts: List of text strings to embed.
        model: Optional pre-loaded model. Loads default if None.

    Returns:
        List of embedding vectors (each is a list of floats).
    """
    if not texts:
        return []

    embedder = model or get_embedding_model()
    embeddings = embedder.embed_documents(texts)

    logger.info(f"Embedded {len(texts)} texts → {len(embeddings)} vectors")

    return embeddings


def embed_query(query: str, model: HuggingFaceEmbeddings | None = None) -> list[float]:
    """
    Embed a single query string.

    WHY a separate function for queries?
        Some embedding models (including bge) use different prefixes for
        queries vs. documents. embed_query() applies the query-specific
        prefix for better retrieval accuracy.

    Args:
        query: The user's question/search query.
        model: Optional pre-loaded model. Loads default if None.

    Returns:
        Single embedding vector as a list of floats.
    """
    embedder = model or get_embedding_model()
    return embedder.embed_query(query)


def validate_embedding_dimension(embedding: list[float]) -> bool:
    """
    Verify that an embedding matches the expected dimension.

    Used as a sanity check to catch model/config mismatches early
    (e.g., loading bge-base instead of bge-small by accident).

    Args:
        embedding: A single embedding vector.

    Returns:
        True if dimension matches EMBEDDING_DIMENSION.
    """
    actual = len(embedding)
    expected = EMBEDDING_DIMENSION

    if actual != expected:
        logger.error(
            f"Embedding dimension mismatch! Expected {expected}, got {actual}. "
            f"Check EMBEDDING_MODEL in settings.py"
        )
        return False

    return True
