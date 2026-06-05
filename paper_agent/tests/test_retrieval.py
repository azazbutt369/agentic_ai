"""
==============================================================================
Tests for the Retrieval Pipeline (Phase 4)
==============================================================================

Covers:
    1. Score normalization — L2 distance → cosine similarity conversion
    2. RetrievalResult — dataclass behavior, auto-populated fields
    3. Similarity search — basic retrieval, relevance ordering
    4. MMR search — diversity in results
    5. Score thresholding — filtering low-relevance noise
    6. Source filtering — restricting to specific papers
    7. Context formatting — prompt-ready context assembly
    8. Utility functions — source summaries, unique sources

HOW TO RUN:
    cd paper_agent
    python -m pytest tests/test_retrieval.py -v
==============================================================================
"""

import sys
import pytest
from pathlib import Path

from langchain_core.documents import Document

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import EMBEDDING_DIMENSION
from embeddings.embedder import get_embedding_model
from vectorstore.faiss_store import create_index
from retrieval.retriever import (
    retrieve,
    format_context,
    get_sources_summary,
    get_unique_sources,
    RetrievalResult,
    normalize_l2_score,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def embedding_model():
    """Load the embedding model once for all tests."""
    return get_embedding_model()


@pytest.fixture(scope="module")
def multi_paper_index(embedding_model):
    """
    Create a FAISS index with chunks from multiple "papers" covering
    different ML topics. This enables testing source filtering and
    relevance ranking across diverse content.
    """
    chunks = [
        # --- Transformer paper chunks ---
        Document(
            page_content=(
                "The Transformer architecture introduces self-attention as the "
                "primary mechanism for sequence modeling, eliminating the need "
                "for recurrent connections entirely."
            ),
            metadata={"source": "transformer.pdf", "page": 1, "chunk_index": 0},
        ),
        Document(
            page_content=(
                "Multi-head attention allows the model to attend to information "
                "from different representation subspaces at different positions. "
                "Each head operates on a projected version of queries, keys, values."
            ),
            metadata={"source": "transformer.pdf", "page": 2, "chunk_index": 1},
        ),
        Document(
            page_content=(
                "Positional encoding is added to the input embeddings to inject "
                "information about the position of tokens in the sequence, since "
                "the Transformer has no built-in notion of order."
            ),
            metadata={"source": "transformer.pdf", "page": 3, "chunk_index": 2},
        ),
        # --- BERT paper chunks ---
        Document(
            page_content=(
                "BERT uses masked language modeling as a pre-training objective, "
                "randomly masking 15% of input tokens and training the model to "
                "predict the masked tokens from context."
            ),
            metadata={"source": "bert.pdf", "page": 1, "chunk_index": 0},
        ),
        Document(
            page_content=(
                "Fine-tuning BERT for downstream tasks requires adding a simple "
                "classification layer on top of the pre-trained model and training "
                "on task-specific labeled data for a few epochs."
            ),
            metadata={"source": "bert.pdf", "page": 4, "chunk_index": 3},
        ),
        # --- CNN paper chunks ---
        Document(
            page_content=(
                "Convolutional neural networks apply learnable filters to extract "
                "spatial features from images. Pooling layers reduce spatial "
                "dimensions while preserving the most important features."
            ),
            metadata={"source": "cnn.pdf", "page": 1, "chunk_index": 0},
        ),
        Document(
            page_content=(
                "Image classification with deep CNNs achieved breakthrough results "
                "on ImageNet, reducing the error rate from 26% to under 4% in "
                "just five years of research progress."
            ),
            metadata={"source": "cnn.pdf", "page": 2, "chunk_index": 1},
        ),
        # --- Reinforcement learning chunks ---
        Document(
            page_content=(
                "Deep reinforcement learning combines neural networks with "
                "reinforcement learning to enable agents to learn policies "
                "directly from high-dimensional sensory inputs like images."
            ),
            metadata={"source": "rl.pdf", "page": 1, "chunk_index": 0},
        ),
    ]

    return create_index(chunks, embedding_model=embedding_model)


# ===========================================================================
# 1. Score Normalization Tests
# ===========================================================================


class TestScoreNormalization:
    """Tests for L2 distance → similarity score conversion."""

    def test_zero_distance_gives_perfect_score(self):
        """L2 distance of 0 (identical vectors) → similarity 1.0."""
        assert normalize_l2_score(0.0) == 1.0

    def test_max_distance_gives_zero_score(self):
        """L2 distance of 2.0 (orthogonal normalized vectors) → similarity 0.0."""
        assert normalize_l2_score(2.0) == 0.0

    def test_intermediate_distance(self):
        """L2 distance of 1.0 → similarity 0.5."""
        assert normalize_l2_score(1.0) == 0.5

    def test_score_is_between_0_and_1(self):
        """All valid L2 distances should produce scores in [0, 1]."""
        for dist in [0.0, 0.1, 0.5, 1.0, 1.5, 2.0]:
            score = normalize_l2_score(dist)
            assert 0.0 <= score <= 1.0, f"Distance {dist} → score {score} out of range"

    def test_negative_distance_clamped(self):
        """Negative distances (floating point artifact) should clamp to max."""
        score = normalize_l2_score(-0.001)
        assert score >= 0.0


# ===========================================================================
# 2. RetrievalResult Tests
# ===========================================================================


class TestRetrievalResult:
    """Tests for the RetrievalResult dataclass."""

    def test_auto_populates_from_metadata(self):
        """Should extract source, page, chunk_index from document metadata."""
        doc = Document(
            page_content="Test content",
            metadata={"source": "paper.pdf", "page": 5, "chunk_index": 3},
        )
        result = RetrievalResult(document=doc, similarity_score=0.85)

        assert result.source == "paper.pdf"
        assert result.page == 5
        assert result.chunk_index == 3

    def test_citation_format(self):
        """Should generate a properly formatted citation string."""
        doc = Document(
            page_content="Content",
            metadata={"source": "attention.pdf", "page": 7, "chunk_index": 0},
        )
        result = RetrievalResult(document=doc, similarity_score=0.9)

        assert result.citation == "[Source: attention.pdf, Page 7]"

    def test_missing_metadata_uses_defaults(self):
        """Should handle missing metadata gracefully."""
        doc = Document(page_content="Content", metadata={})
        result = RetrievalResult(document=doc, similarity_score=0.5)

        assert result.source == "unknown"
        assert result.page == 0
        assert result.chunk_index == 0
        assert "unknown" in result.citation


# ===========================================================================
# 3. Similarity Search Tests
# ===========================================================================


class TestSimilaritySearch:
    """Tests for the core retrieve() function."""

    def test_returns_retrieval_results(self, multi_paper_index):
        """Should return a list of RetrievalResult objects."""
        results = retrieve(
            multi_paper_index,
            "How does attention work?",
            use_mmr=False,
            threshold=0.0,
        )

        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_results_have_scores(self, multi_paper_index):
        """Each result should have a similarity score between 0 and 1."""
        results = retrieve(
            multi_paper_index,
            "What is self-attention?",
            use_mmr=False,
            threshold=0.0,
        )

        for r in results:
            assert 0.0 <= r.similarity_score <= 1.0, (
                f"Score {r.similarity_score} out of range"
            )

    def test_results_sorted_by_score(self, multi_paper_index):
        """Results should be sorted by similarity score (highest first)."""
        results = retrieve(
            multi_paper_index,
            "Transformer self-attention mechanism",
            use_mmr=False,
            threshold=0.0,
        )

        scores = [r.similarity_score for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Results not sorted: {scores}"
        )

    def test_relevance_ranking(self, multi_paper_index):
        """Transformer query should rank Transformer chunks highest."""
        results = retrieve(
            multi_paper_index,
            "How does the Transformer use self-attention?",
            k=8,
            use_mmr=False,
            threshold=0.0,
        )

        # Top result should be from transformer.pdf
        assert results[0].source == "transformer.pdf", (
            f"Expected top result from transformer.pdf, got {results[0].source}"
        )

    def test_k_limits_results(self, multi_paper_index):
        """Should return at most k results."""
        results = retrieve(
            multi_paper_index,
            "neural networks",
            k=2,
            use_mmr=False,
            threshold=0.0,
        )

        assert len(results) <= 2

    def test_results_have_citations(self, multi_paper_index):
        """Each result should have a formatted citation string."""
        results = retrieve(
            multi_paper_index,
            "attention",
            use_mmr=False,
            threshold=0.0,
        )

        for r in results:
            assert r.citation.startswith("[Source:")
            assert "Page" in r.citation

    def test_results_preserve_metadata(self, multi_paper_index):
        """Retrieved documents should have complete metadata."""
        results = retrieve(
            multi_paper_index,
            "masked language modeling",
            use_mmr=False,
            threshold=0.0,
        )

        for r in results:
            assert r.source != ""
            assert r.page > 0
            assert r.document.page_content != ""


# ===========================================================================
# 4. MMR Search Tests
# ===========================================================================


class TestMMRSearch:
    """Tests for MMR (Maximal Marginal Relevance) search."""

    def test_mmr_returns_results(self, multi_paper_index):
        """MMR search should return results."""
        results = retrieve(
            multi_paper_index,
            "attention mechanism in neural networks",
            use_mmr=True,
            threshold=0.0,
        )

        assert len(results) > 0

    def test_mmr_promotes_diversity(self, multi_paper_index):
        """MMR should return results from multiple sources when possible."""
        results = retrieve(
            multi_paper_index,
            "deep learning models and architectures",
            k=4,
            use_mmr=True,
            threshold=0.0,
        )

        sources = [r.source for r in results]
        unique_sources = set(sources)

        # With 4 results across 4 papers, MMR should pull from >= 2 sources
        assert len(unique_sources) >= 2, (
            f"Expected diverse sources but got only {unique_sources}"
        )

    def test_mmr_results_are_retrieval_results(self, multi_paper_index):
        """MMR results should be proper RetrievalResult objects."""
        results = retrieve(
            multi_paper_index,
            "test query",
            use_mmr=True,
            threshold=0.0,
        )

        assert all(isinstance(r, RetrievalResult) for r in results)
        for r in results:
            assert hasattr(r, "similarity_score")
            assert hasattr(r, "citation")


# ===========================================================================
# 5. Score Thresholding Tests
# ===========================================================================


class TestScoreThresholding:
    """Tests for filtering low-relevance results."""

    def test_high_threshold_filters_results(self, multi_paper_index):
        """A very high threshold should return fewer or no results."""
        all_results = retrieve(
            multi_paper_index,
            "attention",
            k=8,
            use_mmr=False,
            threshold=0.0,
        )

        filtered_results = retrieve(
            multi_paper_index,
            "attention",
            k=8,
            use_mmr=False,
            threshold=0.95,  # Very strict
        )

        assert len(filtered_results) <= len(all_results)

    def test_zero_threshold_keeps_all(self, multi_paper_index):
        """Threshold of 0.0 should keep all results."""
        results = retrieve(
            multi_paper_index,
            "anything",
            k=4,
            use_mmr=False,
            threshold=0.0,
        )

        assert len(results) == 4  # Should get exactly k results

    def test_filtered_results_above_threshold(self, multi_paper_index):
        """All returned results should be above the threshold."""
        threshold = 0.3
        results = retrieve(
            multi_paper_index,
            "Transformer model",
            k=8,
            use_mmr=False,
            threshold=threshold,
        )

        for r in results:
            assert r.similarity_score >= threshold, (
                f"Result with score {r.similarity_score} below threshold {threshold}"
            )


# ===========================================================================
# 6. Source Filtering Tests
# ===========================================================================


class TestSourceFiltering:
    """Tests for restricting retrieval to specific papers."""

    def test_source_filter_restricts_results(self, multi_paper_index):
        """Should only return results from the specified source."""
        results = retrieve(
            multi_paper_index,
            "neural network architecture",
            k=8,
            use_mmr=False,
            threshold=0.0,
            source_filter="transformer.pdf",
        )

        for r in results:
            assert r.source == "transformer.pdf", (
                f"Expected transformer.pdf but got {r.source}"
            )

    def test_source_filter_nonexistent_returns_empty(self, multi_paper_index):
        """Filtering by a non-existent source should return empty."""
        results = retrieve(
            multi_paper_index,
            "anything",
            k=4,
            use_mmr=False,
            threshold=0.0,
            source_filter="nonexistent.pdf",
        )

        assert len(results) == 0

    def test_no_source_filter_returns_mixed(self, multi_paper_index):
        """Without source filter, results should come from multiple papers."""
        results = retrieve(
            multi_paper_index,
            "deep learning neural networks",
            k=8,
            use_mmr=False,
            threshold=0.0,
        )

        sources = set(r.source for r in results)
        assert len(sources) > 1, "Expected results from multiple sources"


# ===========================================================================
# 7. Context Formatting Tests
# ===========================================================================


class TestContextFormatting:
    """Tests for format_context() and utility functions."""

    def test_format_context_basic(self):
        """Should format results into a readable context string."""
        results = [
            RetrievalResult(
                document=Document(
                    page_content="Chunk one content.",
                    metadata={"source": "a.pdf", "page": 1, "chunk_index": 0},
                ),
                similarity_score=0.9,
            ),
            RetrievalResult(
                document=Document(
                    page_content="Chunk two content.",
                    metadata={"source": "b.pdf", "page": 3, "chunk_index": 1},
                ),
                similarity_score=0.7,
            ),
        ]

        context = format_context(results)

        assert "[Source: a.pdf, Page 1]" in context
        assert "[Source: b.pdf, Page 3]" in context
        assert "Chunk one content." in context
        assert "Chunk two content." in context

    def test_format_context_empty(self):
        """Should return fallback message for empty results."""
        context = format_context([])
        assert "No relevant context found" in context

    def test_format_context_preserves_order(self):
        """Context chunks should appear in the order given."""
        results = [
            RetrievalResult(
                document=Document(
                    page_content="FIRST chunk.",
                    metadata={"source": "a.pdf", "page": 1, "chunk_index": 0},
                ),
                similarity_score=0.9,
            ),
            RetrievalResult(
                document=Document(
                    page_content="SECOND chunk.",
                    metadata={"source": "a.pdf", "page": 2, "chunk_index": 1},
                ),
                similarity_score=0.8,
            ),
        ]

        context = format_context(results)
        assert context.index("FIRST") < context.index("SECOND")


# ===========================================================================
# 8. Utility Function Tests
# ===========================================================================


class TestUtilities:
    """Tests for get_sources_summary() and get_unique_sources()."""

    def test_sources_summary(self):
        """Should extract source summaries from results."""
        results = [
            RetrievalResult(
                document=Document(
                    page_content="Content here that is long enough for a preview.",
                    metadata={"source": "paper.pdf", "page": 2, "chunk_index": 0},
                ),
                similarity_score=0.85,
            ),
        ]

        summaries = get_sources_summary(results)

        assert len(summaries) == 1
        assert summaries[0]["source"] == "paper.pdf"
        assert summaries[0]["page"] == 2
        assert summaries[0]["score"] == 0.85
        assert "preview" in summaries[0]

    def test_unique_sources(self):
        """Should return deduplicated source filenames in order."""
        results = [
            RetrievalResult(
                document=Document(
                    page_content="A",
                    metadata={"source": "a.pdf", "page": 1, "chunk_index": 0},
                ),
                similarity_score=0.9,
            ),
            RetrievalResult(
                document=Document(
                    page_content="B",
                    metadata={"source": "b.pdf", "page": 1, "chunk_index": 0},
                ),
                similarity_score=0.8,
            ),
            RetrievalResult(
                document=Document(
                    page_content="C",
                    metadata={"source": "a.pdf", "page": 2, "chunk_index": 1},
                ),
                similarity_score=0.7,
            ),
        ]

        sources = get_unique_sources(results)

        assert sources == ["a.pdf", "b.pdf"]  # Deduplicated, order preserved

    def test_unique_sources_empty(self):
        """Should return empty list for no results."""
        assert get_unique_sources([]) == []

    def test_end_to_end_retrieve_and_format(self, multi_paper_index):
        """Full pipeline: retrieve → format_context → ready for LLM."""
        results = retrieve(
            multi_paper_index,
            "How does BERT use masked language modeling?",
            k=3,
            use_mmr=False,
            threshold=0.0,
        )

        context = format_context(results)

        # Context should be non-empty and contain citations
        assert len(context) > 0
        assert "[Source:" in context

        # Top result should be from BERT paper
        assert results[0].source == "bert.pdf", (
            f"Expected BERT source first, got {results[0].source}"
        )


# ===========================================================================
# Run with: python -m pytest tests/test_retrieval.py -v
# ===========================================================================
