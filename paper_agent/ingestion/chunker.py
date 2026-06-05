"""
==============================================================================
Text Chunker — Split documents into retrieval-ready chunks
==============================================================================

WHY CHUNKING MATTERS:
    LLMs have limited context windows. We can't feed an entire 20-page paper
    into the model — we need to find the RELEVANT parts. Chunking breaks the
    paper into small, self-contained pieces that can be individually embedded
    and retrieved.

    The quality of your chunks DIRECTLY determines retrieval quality:
        - Too large → chunks contain mixed topics, retrieval returns noise
        - Too small → chunks lose context, answers lack coherence
        - Bad boundaries → sentences cut mid-thought, embeddings are noisy

CHUNKING STRATEGY — RecursiveCharacterTextSplitter:
    We use LangChain's RecursiveCharacterTextSplitter because it tries
    to split at SEMANTIC BOUNDARIES in priority order:

        1. "\n\n"  → Paragraph breaks (best — preserves full ideas)
        2. "\n"    → Line breaks (good — section boundaries)
        3. ". "    → Sentence ends (okay — complete sentences)
        4. " "     → Word boundaries (last resort — avoids mid-word splits)

    This is superior to naive fixed-size splitting because academic papers
    have natural paragraph structure. A paragraph ≈ one idea ≈ one good chunk.

WHY 512 TOKENS / 64 OVERLAP:
    - 512 tokens ≈ 1 academic paragraph ≈ sufficient context for a single idea
    - 4 chunks × 512 = 2048 tokens of context, fitting within our 2k window
      alongside system prompt (~200 tokens) and chat history (~400 tokens)
    - 64-token overlap (12.5%) prevents losing information at boundaries
      without excessive duplication in the FAISS index

DESIGN DECISIONS:
    - Metadata is PRESERVED through chunking — each chunk knows its source
      file and page number, enabling citation-aware responses.
    - We add a `chunk_index` to metadata for ordering and debugging.
    - Character-based splitting (not token-based) is used because
      RecursiveCharacterTextSplitter is faster and the approximation
      (1 token ≈ 4 chars) is close enough for our purposes.
==============================================================================
"""

import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS

logger = logging.getLogger(__name__)

# Approximate ratio: 1 token ≈ 4 characters for English text.
# This is a well-established heuristic. GPT-family tokenizers average ~3.5-4.5
# chars per token. For academic text (longer words), ~4.0 is a good estimate.
CHARS_PER_TOKEN = 4


def create_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: list[str] | None = None,
) -> RecursiveCharacterTextSplitter:
    """
    Create a configured RecursiveCharacterTextSplitter.

    Args:
        chunk_size: Target chunk size in TOKENS (converted to chars internally).
        chunk_overlap: Overlap between chunks in TOKENS.
        separators: Priority-ordered list of split characters.

    Returns:
        Configured RecursiveCharacterTextSplitter instance.
    """
    size_tokens = chunk_size or CHUNK_SIZE
    overlap_tokens = chunk_overlap or CHUNK_OVERLAP
    seps = separators or CHUNK_SEPARATORS

    # Convert token counts to character counts for the splitter
    size_chars = size_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    logger.info(
        f"Chunker configured: {size_tokens} tokens ({size_chars} chars), "
        f"{overlap_tokens} token overlap ({overlap_chars} chars)"
    )

    return RecursiveCharacterTextSplitter(
        chunk_size=size_chars,
        chunk_overlap=overlap_chars,
        separators=seps,
        length_function=len,  # Character-based length
        is_separator_regex=False,
        strip_whitespace=True,
    )


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split a list of page-level Documents into smaller retrieval-ready chunks.

    Each output chunk inherits metadata from its parent page (source, page
    number) and gains a `chunk_index` for ordering.

    Args:
        documents: List of cleaned Document objects (from text_cleaner).
        chunk_size: Override chunk size in tokens (default from settings).
        chunk_overlap: Override chunk overlap in tokens (default from settings).

    Returns:
        List of chunked Document objects with enriched metadata.

    Example:
        A 10-page paper with ~300 words/page produces roughly 20-30 chunks
        at 512 tokens each.
    """
    if not documents:
        logger.warning("No documents provided for chunking")
        return []

    splitter = create_splitter(chunk_size, chunk_overlap)

    # Split all documents at once — LangChain handles metadata propagation
    raw_chunks = splitter.split_documents(documents)

    # Enrich metadata with chunk index
    enriched_chunks = []
    for idx, chunk in enumerate(raw_chunks):
        chunk.metadata["chunk_index"] = idx
        enriched_chunks.append(chunk)

    # Log statistics
    total_chars = sum(len(c.page_content) for c in enriched_chunks)
    avg_chars = total_chars // len(enriched_chunks) if enriched_chunks else 0

    logger.info(
        f"Chunked {len(documents)} pages → {len(enriched_chunks)} chunks "
        f"(avg {avg_chars} chars ≈ {avg_chars // CHARS_PER_TOKEN} tokens each)"
    )

    return enriched_chunks


def get_chunk_stats(chunks: list[Document]) -> dict:
    """
    Compute statistics about a set of chunks (useful for debugging/UI).

    Returns:
        Dict with: total_chunks, avg_size_chars, avg_size_tokens,
                    min_size_chars, max_size_chars, sources.
    """
    if not chunks:
        return {"total_chunks": 0}

    sizes = [len(c.page_content) for c in chunks]
    sources = list({c.metadata.get("source", "unknown") for c in chunks})

    return {
        "total_chunks": len(chunks),
        "avg_size_chars": sum(sizes) // len(sizes),
        "avg_size_tokens": sum(sizes) // len(sizes) // CHARS_PER_TOKEN,
        "min_size_chars": min(sizes),
        "max_size_chars": max(sizes),
        "sources": sources,
    }
