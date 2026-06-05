"""
==============================================================================
Tests for the Ingestion Pipeline (Phase 2)
==============================================================================

Covers:
    1. PDF Loader — file validation, text extraction, metadata
    2. Text Cleaner — each cleaning function + the pipeline
    3. Chunker — splitting, metadata preservation, stats
    4. Pipeline — end-to-end orchestration

HOW TO RUN:
    cd paper_agent
    python -m pytest tests/test_ingestion.py -v

NOTE: Tests create a small synthetic PDF on-the-fly using PyMuPDF so
      they don't depend on any external test fixtures.
==============================================================================
"""

import sys
import os
import pytest
from pathlib import Path
from unittest.mock import patch

import fitz  # PyMuPDF — used to CREATE test PDFs programmatically
from langchain_core.documents import Document

# Ensure the project root is on sys.path so imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.pdf_loader import load_pdf, get_pdf_metadata
from ingestion.text_cleaner import (
    fix_hyphenated_line_breaks,
    normalize_whitespace,
    fix_unicode_artifacts,
    remove_headers_footers,
    remove_references_section,
    clean_text,
    clean_documents,
)
from ingestion.chunker import (
    create_splitter,
    chunk_documents,
    get_chunk_stats,
    CHARS_PER_TOKEN,
)


# ===========================================================================
# Fixtures — Create synthetic test PDFs
# ===========================================================================


@pytest.fixture
def sample_pdf(tmp_path) -> Path:
    """
    Create a synthetic 3-page PDF with realistic academic paper content.
    Returns the path to the created PDF.
    """
    pdf_path = tmp_path / "test_paper.pdf"
    doc = fitz.open()

    # Page 1 — Abstract-like content
    page1 = doc.new_page()
    page1.insert_text(
        (72, 72),
        "1\n"  # Page number header (should be stripped)
        "Attention Is All You Need\n\n"
        "Abstract\n\n"
        "The dominant sequence transduction models are based on complex "
        "recurrent or convolutional neural networks that include an encoder "
        "and a decoder. The best performing models also connect the encoder "
        "and decoder through an attention mechanism. We propose a new simple "
        "network architecture, the Transformer, based solely on attention "
        "mechanisms, dispensing with recurrence and convolutions entirely. "
        "Experiments on two machine translation tasks show these models to "
        "be superior in quality while being more parallelizable and requiring "
        "significantly less time to train.\n\n"
        "The Transformer model achieves state-of-the-art results on the "
        "WMT 2014 English-to-German translation task, improving over the "
        "existing best results by over 2 BLEU. On the WMT 2014 English-to-"
        "French translation task, our model establishes a new single-model "
        "state-of-the-art BLEU score of 41.8 after training for 3.5 days.",
        fontsize=10,
    )

    # Page 2 — Method-like content with hyphenation and unicode artifacts
    page2 = doc.new_page()
    page2.insert_text(
        (72, 72),
        "2\n"  # Page number header
        "Multi-Head Attention\n\n"
        "Instead of performing a single attention function with "
        "d_model-dimensional keys, values and queries, we found it bene\ufb01cial "
        "to linearly project the queries, keys and values h times with "
        "different, learned linear projections to d_k, d_k and d_v dimensions, "
        "respectively.\n\n"
        "Multi-head attention allows the model to jointly attend to infor-\n"
        "mation from different representation subspaces at different positions. "
        "With a single attention head, averaging inhibits this.\n\n"
        "The Transformer uses multi-head attention in three different ways. "
        "First, in encoder-decoder attention layers, the queries come from "
        "the previous decoder layer, and the memory keys and values come from "
        "the output of the encoder. This allows every position in the decoder "
        "to attend over all positions in the input sequence.",
        fontsize=10,
    )

    # Page 3 — References section
    page3 = doc.new_page()
    page3.insert_text(
        (72, 72),
        "3\n"
        "Conclusion\n\n"
        "In this work, we presented the Transformer, the first sequence "
        "transduction model based entirely on attention, replacing the "
        "recurrent layers most commonly used in encoder-decoder architectures "
        "with multi-headed self-attention.\n\n"
        "References\n\n"
        "[1] Bahdanau, D., Cho, K., and Bengio, Y. Neural machine "
        "translation by jointly learning to align and translate. ICLR, 2015.\n"
        "[2] Gehring, J., Auli, M., Grangier, D., Yasin, D., and Dauphin, "
        "Y. N. Convolutional sequence to sequence learning. ICML, 2017.\n"
        "[3] Vaswani, A., et al. Attention is all you need. NeurIPS, 2017.",
        fontsize=10,
    )

    doc.save(str(pdf_path))
    doc.close()

    return pdf_path


@pytest.fixture
def empty_pdf(tmp_path) -> Path:
    """Create a PDF with no text content (blank pages)."""
    pdf_path = tmp_path / "empty.pdf"
    doc = fitz.open()
    doc.new_page()  # Blank page
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample Document objects mimicking pdf_loader output."""
    return [
        Document(
            page_content=(
                "The Transformer architecture uses self-attention mechanisms "
                "to process input sequences in parallel. Unlike recurrent "
                "neural networks, Transformers do not require sequential "
                "computation, enabling significantly faster training times. "
                "The key innovation is the multi-head attention mechanism, "
                "which allows the model to focus on different parts of the "
                "input simultaneously. This parallel processing capability "
                "makes Transformers particularly well-suited for modern GPU "
                "hardware, where parallel operations are highly optimized."
            ),
            metadata={"source": "test.pdf", "page": 1, "total_pages": 3},
        ),
        Document(
            page_content=(
                "The attention mechanism computes a weighted sum of values, "
                "where the weights are determined by the compatibility of "
                "queries and keys. Scaled dot-product attention divides by "
                "the square root of the key dimension to prevent the dot "
                "products from growing too large, which would push the "
                "softmax function into regions with extremely small gradients. "
                "Multi-head attention runs multiple attention functions in "
                "parallel, concatenates the results, and projects them with "
                "a learned linear transformation. This allows the model to "
                "attend to information from different representation subspaces."
            ),
            metadata={"source": "test.pdf", "page": 2, "total_pages": 3},
        ),
        Document(
            page_content=(
                "Position-wise feed-forward networks are applied to each "
                "position separately and identically. Each layer consists "
                "of two linear transformations with a ReLU activation in "
                "between. The dimensionality of input and output is 512, "
                "while the inner layer has dimensionality 2048. Residual "
                "connections and layer normalization are employed around "
                "each sub-layer to facilitate training of deep networks."
            ),
            metadata={"source": "test.pdf", "page": 3, "total_pages": 3},
        ),
    ]


# ===========================================================================
# 1. PDF Loader Tests
# ===========================================================================


class TestPDFLoader:
    """Tests for ingestion/pdf_loader.py"""

    def test_load_pdf_returns_documents(self, sample_pdf):
        """Loading a valid PDF should return a list of Document objects."""
        docs = load_pdf(sample_pdf)
        assert isinstance(docs, list)
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_load_pdf_preserves_metadata(self, sample_pdf):
        """Each Document should have source, page, and total_pages metadata."""
        docs = load_pdf(sample_pdf)

        for doc in docs:
            assert "source" in doc.metadata
            assert "page" in doc.metadata
            assert "total_pages" in doc.metadata
            assert doc.metadata["source"] == "test_paper.pdf"
            assert doc.metadata["total_pages"] == 3

    def test_load_pdf_pages_are_1_indexed(self, sample_pdf):
        """Page numbers should be 1-indexed (human-readable)."""
        docs = load_pdf(sample_pdf)
        page_numbers = [d.metadata["page"] for d in docs]
        assert min(page_numbers) >= 1

    def test_load_pdf_extracts_text(self, sample_pdf):
        """Extracted text should contain expected content."""
        docs = load_pdf(sample_pdf)
        all_text = " ".join(d.page_content for d in docs)

        assert "Transformer" in all_text
        assert "attention" in all_text.lower()

    def test_load_pdf_file_not_found(self):
        """Should raise FileNotFoundError for non-existent files."""
        with pytest.raises(FileNotFoundError):
            load_pdf("/nonexistent/path/paper.pdf")

    def test_load_pdf_wrong_extension(self, tmp_path):
        """Should raise ValueError for non-PDF files."""
        txt_file = tmp_path / "paper.txt"
        txt_file.write_text("Not a PDF")

        with pytest.raises(ValueError, match="Expected a .pdf file"):
            load_pdf(txt_file)

    def test_load_pdf_empty_raises_error(self, empty_pdf):
        """Should raise ValueError for PDFs with no extractable text."""
        with pytest.raises(ValueError, match="No extractable text"):
            load_pdf(empty_pdf)

    def test_get_pdf_metadata(self, sample_pdf):
        """Metadata extraction should return expected fields."""
        meta = get_pdf_metadata(sample_pdf)

        assert "title" in meta
        assert "author" in meta
        assert "pages" in meta
        assert meta["pages"] == 3

    def test_get_pdf_metadata_file_not_found(self):
        """Should raise FileNotFoundError for non-existent files."""
        with pytest.raises(FileNotFoundError):
            get_pdf_metadata("/nonexistent/path/paper.pdf")


# ===========================================================================
# 2. Text Cleaner Tests
# ===========================================================================


class TestTextCleaner:
    """Tests for ingestion/text_cleaner.py"""

    # --- Individual cleaning functions ---

    def test_fix_hyphenated_line_breaks(self):
        """Should rejoin words split by hyphenation across lines."""
        text = "This is an atten-\ntion mechanism used in trans-\nformer models."
        result = fix_hyphenated_line_breaks(text)
        assert "attention" in result
        assert "transformer" in result
        assert "atten-\n" not in result

    def test_fix_hyphenated_preserves_real_hyphens(self):
        """Should NOT remove hyphens that are part of real compound words."""
        text = "This is a well-known self-attention mechanism.\n"
        result = fix_hyphenated_line_breaks(text)
        assert "well-known" in result
        assert "self-attention" in result

    def test_normalize_whitespace_collapses_spaces(self):
        """Should collapse multiple spaces into one."""
        text = "Hello    world   this    is   text"
        result = normalize_whitespace(text)
        assert "  " not in result
        assert result == "Hello world this is text"

    def test_normalize_whitespace_collapses_newlines(self):
        """Should collapse 3+ newlines into 2 (paragraph break)."""
        text = "Paragraph one.\n\n\n\n\nParagraph two."
        result = normalize_whitespace(text)
        assert "\n\n\n" not in result
        assert "Paragraph one.\n\nParagraph two." == result

    def test_normalize_whitespace_strips_trailing(self):
        """Should remove trailing whitespace on each line."""
        text = "Hello world   \nNext line   "
        result = normalize_whitespace(text)
        assert not any(line.endswith(" ") for line in result.split("\n"))

    def test_fix_unicode_ligatures(self):
        """Should replace common Unicode ligatures with ASCII equivalents."""
        text = "This is bene\ufb01cial for ef\ufb01cient processing."
        result = fix_unicode_artifacts(text)
        assert "beneficial" in result
        assert "efficient" in result
        assert "\ufb01" not in result

    def test_fix_unicode_dashes(self):
        """Should normalize en/em dashes to hyphens."""
        text = "state\u2013of\u2013the\u2014art"
        result = fix_unicode_artifacts(text)
        assert result == "state-of-the-art"

    def test_fix_unicode_quotes(self):
        """Should normalize smart quotes to standard ASCII quotes."""
        text = "\u201cHello\u201d said \u2018world\u2019"
        result = fix_unicode_artifacts(text)
        assert '"Hello"' in result
        assert "'world'" in result

    def test_remove_headers_footers_strips_page_numbers(self):
        """Should remove short numeric lines at top/bottom of page."""
        text = (
            "42\n"
            "Main content of the page goes here.\n"
            "This is important text.\n"
            "More content on this page.\n"
            "Even more content here.\n"
            "7"
        )
        result = remove_headers_footers(text)
        assert "Main content" in result
        assert result.strip().startswith("Main content")

    def test_remove_headers_footers_preserves_short_content(self):
        """Should not strip content from very short pages."""
        text = "Short\npage"
        result = remove_headers_footers(text)
        assert result == text  # Too short, nothing removed

    def test_remove_references_section(self):
        """Should remove everything after 'References' header."""
        text = (
            "Main content here.\n\n"
            "Conclusion text.\n\n"
            "References\n\n"
            "[1] Some paper citation.\n"
            "[2] Another paper citation."
        )
        result = remove_references_section(text)
        assert "Main content" in result
        assert "Conclusion" in result
        assert "[1]" not in result
        assert "[2]" not in result

    def test_remove_references_handles_ALLCAPS(self):
        """Should handle REFERENCES in all caps."""
        text = "Main content.\n\nRESEARENCES\n\n[1] Citation."
        # Note: "RESERENCES" won't match but "REFERENCES" will
        text2 = "Main content.\n\nREFERENCES\n\n[1] Citation."
        result = remove_references_section(text2)
        assert "[1]" not in result

    def test_remove_references_no_section_preserves_text(self):
        """Should return text unchanged if no references section found."""
        text = "Just normal content without any references section."
        result = remove_references_section(text)
        assert result == text

    # --- Full pipeline ---

    def test_clean_text_pipeline(self):
        """Full pipeline should apply all cleaning steps."""
        messy_text = (
            "42\n"
            "This is bene\ufb01cial for atten-\n"
            "tion    mechanisms   in modern    NLP.\n"
            "15"
        )
        result = clean_text(messy_text)
        assert "beneficial" in result
        assert "attention" in result
        assert "  " not in result  # No double spaces

    def test_clean_text_empty_input(self):
        """Should handle empty input gracefully."""
        assert clean_text("") == ""
        assert clean_text("   ") == ""

    def test_clean_documents_preserves_metadata(self):
        """Cleaning should preserve document metadata."""
        docs = [
            Document(
                page_content="This is valid content with enough characters to pass.",
                metadata={"source": "test.pdf", "page": 1},
            )
        ]
        cleaned = clean_documents(docs)
        assert len(cleaned) == 1
        assert cleaned[0].metadata["source"] == "test.pdf"
        assert cleaned[0].metadata["page"] == 1

    def test_clean_documents_removes_empty_pages(self):
        """Should remove pages that become empty after cleaning."""
        docs = [
            Document(
                page_content="Valid content with sufficient length for testing.",
                metadata={"source": "test.pdf", "page": 1},
            ),
            Document(
                page_content="   ",  # Will be empty after cleaning
                metadata={"source": "test.pdf", "page": 2},
            ),
        ]
        cleaned = clean_documents(docs)
        assert len(cleaned) == 1
        assert cleaned[0].metadata["page"] == 1


# ===========================================================================
# 3. Chunker Tests
# ===========================================================================


class TestChunker:
    """Tests for ingestion/chunker.py"""

    def test_create_splitter_default_config(self):
        """Should create a splitter with default settings."""
        splitter = create_splitter()
        assert splitter is not None
        # Check that chunk_size was converted to chars
        assert splitter._chunk_size == 512 * CHARS_PER_TOKEN

    def test_create_splitter_custom_config(self):
        """Should accept custom chunk size and overlap."""
        splitter = create_splitter(chunk_size=256, chunk_overlap=32)
        assert splitter._chunk_size == 256 * CHARS_PER_TOKEN
        assert splitter._chunk_overlap == 32 * CHARS_PER_TOKEN

    def test_chunk_documents_produces_chunks(self, sample_documents):
        """Should split documents into smaller chunks."""
        # Use small chunk size to force splitting
        chunks = chunk_documents(sample_documents, chunk_size=64, chunk_overlap=8)
        assert len(chunks) > len(sample_documents)

    def test_chunk_documents_preserves_metadata(self, sample_documents):
        """Each chunk should inherit source and page metadata."""
        chunks = chunk_documents(sample_documents, chunk_size=64, chunk_overlap=8)

        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "page" in chunk.metadata
            assert "total_pages" in chunk.metadata
            assert chunk.metadata["source"] == "test.pdf"

    def test_chunk_documents_adds_chunk_index(self, sample_documents):
        """Each chunk should have a unique chunk_index in metadata."""
        chunks = chunk_documents(sample_documents, chunk_size=64, chunk_overlap=8)

        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))  # Sequential 0, 1, 2, ...

    def test_chunk_documents_respects_size(self, sample_documents):
        """Chunks should not exceed the configured chunk size (in chars)."""
        target_size = 128  # tokens
        chunks = chunk_documents(
            sample_documents, chunk_size=target_size, chunk_overlap=16
        )

        max_chars = target_size * CHARS_PER_TOKEN
        for chunk in chunks:
            # Allow small overflow from separator logic
            assert len(chunk.page_content) <= max_chars + 50, (
                f"Chunk too large: {len(chunk.page_content)} chars "
                f"(max: {max_chars})"
            )

    def test_chunk_documents_empty_input(self):
        """Should handle empty input gracefully."""
        chunks = chunk_documents([])
        assert chunks == []

    def test_chunk_documents_single_small_doc(self):
        """A document smaller than chunk_size should remain as one chunk."""
        small_doc = [
            Document(
                page_content="Short text.",
                metadata={"source": "test.pdf", "page": 1, "total_pages": 1},
            )
        ]
        chunks = chunk_documents(small_doc, chunk_size=512, chunk_overlap=64)
        assert len(chunks) == 1
        assert chunks[0].page_content == "Short text."

    def test_get_chunk_stats(self, sample_documents):
        """Should compute correct statistics about chunks."""
        chunks = chunk_documents(sample_documents, chunk_size=64, chunk_overlap=8)
        stats = get_chunk_stats(chunks)

        assert stats["total_chunks"] == len(chunks)
        assert stats["avg_size_chars"] > 0
        assert stats["avg_size_tokens"] > 0
        assert stats["min_size_chars"] <= stats["max_size_chars"]
        assert "test.pdf" in stats["sources"]

    def test_get_chunk_stats_empty(self):
        """Should handle empty chunk list."""
        stats = get_chunk_stats([])
        assert stats["total_chunks"] == 0


# ===========================================================================
# 4. End-to-End Pipeline Tests
# ===========================================================================


class TestIngestionPipeline:
    """Tests for ingestion/pipeline.py — full pipeline integration."""

    def test_ingest_pdf_end_to_end(self, sample_pdf):
        """Full pipeline should produce chunks with metadata and stats."""
        from ingestion.pipeline import ingest_pdf

        result = ingest_pdf(sample_pdf)

        # Check result structure
        assert "chunks" in result
        assert "metadata" in result
        assert "stats" in result

        # Check chunks
        assert len(result["chunks"]) > 0
        assert all(isinstance(c, Document) for c in result["chunks"])

        # Check metadata
        assert result["metadata"]["pages"] == 3

        # Check stats
        assert result["stats"]["total_chunks"] == len(result["chunks"])

    def test_ingest_pdf_with_reference_stripping(self, sample_pdf):
        """Pipeline with strip_references=True should produce fewer chunks."""
        from ingestion.pipeline import ingest_pdf

        result_with_refs = ingest_pdf(sample_pdf, strip_references=False)
        result_without_refs = ingest_pdf(sample_pdf, strip_references=True)

        # With references stripped, total text should be smaller
        text_with = sum(len(c.page_content) for c in result_with_refs["chunks"])
        text_without = sum(
            len(c.page_content) for c in result_without_refs["chunks"]
        )

        # Text without references should be <= text with references
        assert text_without <= text_with

    def test_ingest_pdf_custom_chunk_size(self, sample_pdf):
        """Custom chunk size should produce different chunk counts."""
        from ingestion.pipeline import ingest_pdf

        result_small = ingest_pdf(sample_pdf, chunk_size=64)
        result_large = ingest_pdf(sample_pdf, chunk_size=1024)

        # Smaller chunks → more chunks
        assert len(result_small["chunks"]) >= len(result_large["chunks"])

    def test_ingest_pdf_chunks_have_full_metadata(self, sample_pdf):
        """Every chunk from the pipeline should have complete metadata."""
        from ingestion.pipeline import ingest_pdf

        result = ingest_pdf(sample_pdf)

        for chunk in result["chunks"]:
            meta = chunk.metadata
            assert "source" in meta
            assert "page" in meta
            assert "total_pages" in meta
            assert "chunk_index" in meta
            assert meta["source"] == "test_paper.pdf"

    def test_save_uploaded_pdf(self, tmp_path):
        """Should save uploaded bytes to the papers directory."""
        from ingestion.pipeline import save_uploaded_pdf

        # Create fake PDF bytes
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test content for upload.")
        pdf_bytes = doc.tobytes()
        doc.close()

        # Temporarily override PAPERS_DIR
        with patch("ingestion.pipeline.PAPERS_DIR", tmp_path):
            saved_path = save_uploaded_pdf("uploaded_test.pdf", pdf_bytes)

        assert saved_path.exists()
        assert saved_path.name == "uploaded_test.pdf"
        assert saved_path.stat().st_size > 0


# ===========================================================================
# Run with: python -m pytest tests/test_ingestion.py -v
# ===========================================================================
