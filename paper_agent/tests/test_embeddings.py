"""
==============================================================================
Tests for Embeddings & FAISS Vector Store (Phase 3)
==============================================================================

Covers:
    1. Embedder — model loading, text embedding, query embedding, dimensions
    2. FAISS Store — index creation, save/load, incremental add, search, stats

HOW TO RUN:
    cd paper_agent
    python -m pytest tests/test_embeddings.py -v

NOTE: These tests load the actual bge-small-en-v1.5 model (~130 MB).
      First run may take 30-60s to download the model. Subsequent runs
      use the cached model and complete in seconds.
==============================================================================
"""

import sys
import numbers
import pytest
from pathlib import Path

from langchain_core.documents import Document

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import EMBEDDING_DIMENSION
from embeddings.embedder import (
    get_embedding_model,
    embed_texts,
    embed_query,
    validate_embedding_dimension,
    _detect_device,
)
from vectorstore.faiss_store import (
    create_index,
    save_index,
    load_index,
    add_documents,
    delete_index,
    similarity_search,
    get_index_stats,
    index_exists,
    DEFAULT_INDEX_NAME,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def embedding_model():
    """
    Load the embedding model ONCE for all tests in this module.

    scope="module" means the model is loaded once and shared across all
    test methods. This is critical because loading takes ~2 seconds and
    we don't want to pay that cost 20+ times.
    """
    return get_embedding_model()


@pytest.fixture
def sample_chunks() -> list[Document]:
    """Create sample Document chunks for index testing."""
    return [
        Document(
            page_content=(
                "The Transformer architecture uses self-attention mechanisms "
                "to process input sequences in parallel, enabling significantly "
                "faster training compared to recurrent neural networks."
            ),
            metadata={"source": "transformers.pdf", "page": 1, "chunk_index": 0},
        ),
        Document(
            page_content=(
                "BERT is a pre-trained language model that uses bidirectional "
                "transformers. It achieves state-of-the-art results on eleven "
                "natural language processing benchmarks."
            ),
            metadata={"source": "bert.pdf", "page": 1, "chunk_index": 0},
        ),
        Document(
            page_content=(
                "Convolutional neural networks are primarily used for image "
                "recognition and computer vision tasks. They use convolutional "
                "layers to extract spatial features from input images."
            ),
            metadata={"source": "cnn.pdf", "page": 1, "chunk_index": 0},
        ),
        Document(
            page_content=(
                "Gradient descent is an optimization algorithm used to minimize "
                "the loss function in machine learning models. Stochastic "
                "gradient descent processes mini-batches for efficiency."
            ),
            metadata={"source": "optimization.pdf", "page": 3, "chunk_index": 2},
        ),
        Document(
            page_content=(
                "The attention mechanism computes a weighted sum of values "
                "where weights are determined by query-key compatibility. "
                "Multi-head attention runs multiple attention functions in parallel."
            ),
            metadata={"source": "transformers.pdf", "page": 2, "chunk_index": 1},
        ),
    ]


@pytest.fixture
def extra_chunks() -> list[Document]:
    """Additional chunks for incremental add testing."""
    return [
        Document(
            page_content=(
                "Reinforcement learning trains agents to make decisions by "
                "maximizing cumulative reward through trial and error interaction "
                "with an environment."
            ),
            metadata={"source": "rl.pdf", "page": 1, "chunk_index": 0},
        ),
        Document(
            page_content=(
                "Generative adversarial networks consist of a generator and "
                "a discriminator that compete against each other to produce "
                "increasingly realistic synthetic data."
            ),
            metadata={"source": "gan.pdf", "page": 1, "chunk_index": 0},
        ),
    ]


# ===========================================================================
# 1. Embedder Tests
# ===========================================================================


class TestEmbedder:
    """Tests for embeddings/embedder.py"""

    def test_detect_device_returns_valid_string(self):
        """Device detection should return 'cpu' or 'cuda'."""
        device = _detect_device()
        assert device in ("cpu", "cuda")

    def test_get_embedding_model_returns_model(self, embedding_model):
        """Should return a HuggingFaceEmbeddings instance."""
        assert embedding_model is not None
        assert hasattr(embedding_model, "embed_documents")
        assert hasattr(embedding_model, "embed_query")

    def test_get_embedding_model_is_singleton(self):
        """Calling get_embedding_model() twice should return the same object."""
        model1 = get_embedding_model()
        model2 = get_embedding_model()
        assert model1 is model2  # Same object in memory

    def test_embed_texts_returns_vectors(self, embedding_model):
        """Should return a list of embedding vectors."""
        texts = ["Hello world", "Machine learning is great"]
        vectors = embed_texts(texts, model=embedding_model)

        assert len(vectors) == 2
        assert all(isinstance(v, list) for v in vectors)
        assert all(len(v) == EMBEDDING_DIMENSION for v in vectors)

    def test_embed_texts_values_are_floats(self, embedding_model):
        """Each element in the embedding vector should be a float."""
        vectors = embed_texts(["Test text"], model=embedding_model)
        assert all(isinstance(x, float) for x in vectors[0])

    def test_embed_texts_empty_input(self, embedding_model):
        """Should handle empty input gracefully."""
        vectors = embed_texts([], model=embedding_model)
        assert vectors == []

    def test_embed_query_returns_single_vector(self, embedding_model):
        """Should return a single embedding vector for a query."""
        vector = embed_query("What is a Transformer?", model=embedding_model)

        assert isinstance(vector, list)
        assert len(vector) == EMBEDDING_DIMENSION

    def test_embed_query_returns_correct_dimension(self, embedding_model):
        """Query embedding should have the correct dimension and be usable for search.

        NOTE: HuggingFaceEmbeddings does NOT apply different prefixes for
        queries vs documents by default, so they produce identical vectors.
        This is fine — the retrieval quality is still good because both
        are embedded by the same model in the same space.
        """
        text = "The Transformer uses multi-head attention"
        doc_vec = embed_texts([text], model=embedding_model)[0]
        query_vec = embed_query(text, model=embedding_model)

        # Both should have the correct dimension
        assert len(doc_vec) == EMBEDDING_DIMENSION
        assert len(query_vec) == EMBEDDING_DIMENSION

        # Cosine similarity between identical text should be very high
        sim = sum(d * q for d, q in zip(doc_vec, query_vec))
        assert sim > 0.95, f"Expected high similarity for identical text, got {sim:.4f}"

    def test_similar_texts_have_close_embeddings(self, embedding_model):
        """Semantically similar texts should have closer embeddings."""
        text_a = "The Transformer model uses self-attention"
        text_b = "Self-attention is the core mechanism of Transformers"
        text_c = "I went to the grocery store to buy apples"

        vecs = embed_texts([text_a, text_b, text_c], model=embedding_model)

        # Cosine similarity via dot product (embeddings are L2-normalized)
        sim_ab = sum(a * b for a, b in zip(vecs[0], vecs[1]))
        sim_ac = sum(a * c for a, c in zip(vecs[0], vecs[2]))

        # Transformer texts should be more similar to each other
        # than either is to the grocery store text
        assert sim_ab > sim_ac, (
            f"Expected sim(transformer, transformer) > sim(transformer, grocery) "
            f"but got {sim_ab:.4f} vs {sim_ac:.4f}"
        )

    def test_validate_embedding_dimension_correct(self, embedding_model):
        """Should return True for correct dimension."""
        vector = embed_query("test", model=embedding_model)
        assert validate_embedding_dimension(vector) is True

    def test_validate_embedding_dimension_wrong(self):
        """Should return False for wrong dimension."""
        fake_vector = [0.1] * 100  # Wrong dimension
        assert validate_embedding_dimension(fake_vector) is False


# ===========================================================================
# 2. FAISS Store Tests
# ===========================================================================


class TestFAISSStore:
    """Tests for vectorstore/faiss_store.py"""

    def test_create_index_from_documents(self, sample_chunks, embedding_model):
        """Should create a FAISS index from documents."""
        index = create_index(sample_chunks, embedding_model=embedding_model)

        assert index is not None
        assert index.index.ntotal == len(sample_chunks)
        assert index.index.d == EMBEDDING_DIMENSION

    def test_create_index_empty_raises_error(self, embedding_model):
        """Should raise ValueError for empty document list."""
        with pytest.raises(ValueError, match="Cannot create index from empty"):
            create_index([], embedding_model=embedding_model)

    def test_save_and_load_index(self, sample_chunks, embedding_model, tmp_path):
        """Should persist index to disk and reload it correctly."""
        # Create and save
        index = create_index(sample_chunks, embedding_model=embedding_model)
        save_index(index, path=tmp_path)

        # Verify files exist
        assert (tmp_path / f"{DEFAULT_INDEX_NAME}.faiss").exists()
        assert (tmp_path / f"{DEFAULT_INDEX_NAME}.pkl").exists()

        # Load and verify
        loaded = load_index(path=tmp_path, embedding_model=embedding_model)
        assert loaded.index.ntotal == index.index.ntotal
        assert loaded.index.d == index.index.d

    def test_load_index_not_found(self, tmp_path, embedding_model):
        """Should raise FileNotFoundError for missing index."""
        with pytest.raises(FileNotFoundError, match="FAISS index not found"):
            load_index(path=tmp_path / "nonexistent", embedding_model=embedding_model)

    def test_add_documents_incremental(
        self, sample_chunks, extra_chunks, embedding_model
    ):
        """Adding documents should increase the index size."""
        index = create_index(sample_chunks, embedding_model=embedding_model)
        original_count = index.index.ntotal

        add_documents(index, extra_chunks)

        assert index.index.ntotal == original_count + len(extra_chunks)

    def test_add_documents_empty(self, sample_chunks, embedding_model):
        """Adding empty list should not change the index."""
        index = create_index(sample_chunks, embedding_model=embedding_model)
        original_count = index.index.ntotal

        add_documents(index, [])

        assert index.index.ntotal == original_count

    def test_similarity_search_returns_results(
        self, sample_chunks, embedding_model
    ):
        """Search should return documents with scores."""
        index = create_index(sample_chunks, embedding_model=embedding_model)

        results = similarity_search(index, "What is the Transformer?", k=3)

        assert len(results) == 3
        assert all(isinstance(r, tuple) for r in results)
        assert all(isinstance(r[0], Document) for r in results)
        # FAISS returns numpy.float32 scores, use numbers.Real for flexible check
        assert all(isinstance(r[1], numbers.Real) for r in results)

    def test_similarity_search_relevance(self, sample_chunks, embedding_model):
        """Transformer-related query should rank Transformer chunks higher."""
        index = create_index(sample_chunks, embedding_model=embedding_model)

        results = similarity_search(
            index, "How does self-attention work in Transformers?", k=5
        )

        # The top result should be from transformers.pdf (not cnn.pdf or optimization.pdf)
        top_source = results[0][0].metadata["source"]
        assert top_source == "transformers.pdf", (
            f"Expected top result from transformers.pdf but got {top_source}"
        )

    def test_similarity_search_k_limit(self, sample_chunks, embedding_model):
        """Should return at most k results."""
        index = create_index(sample_chunks, embedding_model=embedding_model)

        results = similarity_search(index, "test query", k=2)
        assert len(results) == 2

    def test_similarity_search_preserves_metadata(
        self, sample_chunks, embedding_model
    ):
        """Search results should have complete metadata."""
        index = create_index(sample_chunks, embedding_model=embedding_model)

        results = similarity_search(index, "neural networks", k=3)

        for doc, score in results:
            assert "source" in doc.metadata
            assert "page" in doc.metadata
            assert "chunk_index" in doc.metadata

    def test_get_index_stats(self, sample_chunks, embedding_model):
        """Should return correct index statistics."""
        index = create_index(sample_chunks, embedding_model=embedding_model)
        stats = get_index_stats(index)

        assert stats["total_vectors"] == len(sample_chunks)
        assert stats["dimension"] == EMBEDDING_DIMENSION
        assert "index_type" in stats

    def test_index_exists_true(self, sample_chunks, embedding_model, tmp_path):
        """Should return True when index files exist."""
        index = create_index(sample_chunks, embedding_model=embedding_model)
        save_index(index, path=tmp_path)

        assert index_exists(path=tmp_path) is True

    def test_index_exists_false(self, tmp_path):
        """Should return False when index files don't exist."""
        assert index_exists(path=tmp_path / "nonexistent") is False

    def test_delete_index(self, sample_chunks, embedding_model, tmp_path):
        """Should remove index files from disk."""
        index = create_index(sample_chunks, embedding_model=embedding_model)
        save_index(index, path=tmp_path)

        assert index_exists(path=tmp_path) is True

        deleted = delete_index(path=tmp_path)

        assert deleted is True
        assert index_exists(path=tmp_path) is False

    def test_delete_index_nonexistent(self, tmp_path):
        """Should return False when trying to delete non-existent index."""
        result = delete_index(path=tmp_path / "nonexistent")
        assert result is False

    def test_save_load_preserves_search_quality(
        self, sample_chunks, embedding_model, tmp_path
    ):
        """Search results should be identical before and after save/load."""
        index = create_index(sample_chunks, embedding_model=embedding_model)
        query = "attention mechanism"

        # Search before save
        results_before = similarity_search(index, query, k=3)

        # Save and reload
        save_index(index, path=tmp_path)
        loaded = load_index(path=tmp_path, embedding_model=embedding_model)

        # Search after load
        results_after = similarity_search(loaded, query, k=3)

        # Same documents should be returned in the same order
        sources_before = [r[0].metadata["source"] for r in results_before]
        sources_after = [r[0].metadata["source"] for r in results_after]
        assert sources_before == sources_after

        # Scores should be very close (floating point allows tiny differences)
        for (_, score_b), (_, score_a) in zip(results_before, results_after):
            assert abs(score_b - score_a) < 1e-5

    def test_incremental_add_then_search(
        self, sample_chunks, extra_chunks, embedding_model
    ):
        """New documents added incrementally should be searchable."""
        index = create_index(sample_chunks, embedding_model=embedding_model)

        # Add RL and GAN documents
        add_documents(index, extra_chunks)

        # Search for RL content
        results = similarity_search(
            index, "reinforcement learning reward agent", k=3
        )

        # The RL document should be in the top results
        sources = [r[0].metadata["source"] for r in results]
        assert "rl.pdf" in sources, (
            f"Expected 'rl.pdf' in search results but got {sources}"
        )


# ===========================================================================
# Run with: python -m pytest tests/test_embeddings.py -v
# ===========================================================================
