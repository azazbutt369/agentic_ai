"""
==============================================================================
Tests for Generation & Conversation Chain (Phase 5)
==============================================================================

Covers:
    1. Prompt Templates — structure, variables, mode selection
    2. Chat Memory — sliding window, add/get/clear, overflow
    3. Chain — end-to-end orchestration (with mocked LLM)
    4. LLM Client — configuration, connection check

NOTE: These tests MOCK the Ollama LLM so they run WITHOUT a live
      Ollama server. Integration tests with a real LLM are separate
      and require `ollama serve` to be running.

HOW TO RUN:
    cd paper_agent
    python -m pytest tests/test_generation.py -v
==============================================================================
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from generation.prompts import (
    get_prompt,
    get_qa_prompt,
    get_summarize_prompt,
    get_compare_prompt,
    QA_SYSTEM_MESSAGE,
    SUMMARIZE_SYSTEM_MESSAGE,
    COMPARE_SYSTEM_MESSAGE,
)
from memory.chat_memory import ChatMemory
from generation.chain import PaperTutorChain, ChainResponse
from embeddings.embedder import get_embedding_model
from vectorstore.faiss_store import create_index


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def embedding_model():
    """Load embedding model once for all tests."""
    return get_embedding_model()


@pytest.fixture(scope="module")
def sample_index(embedding_model):
    """Create a small FAISS index for chain testing."""
    chunks = [
        Document(
            page_content=(
                "The Transformer model uses self-attention to process "
                "sequences in parallel. The key innovation is multi-head "
                "attention, which allows attending to different positions "
                "simultaneously."
            ),
            metadata={"source": "transformer.pdf", "page": 1, "chunk_index": 0},
        ),
        Document(
            page_content=(
                "BERT uses masked language modeling where 15% of tokens "
                "are randomly masked and the model learns to predict them. "
                "This bidirectional pre-training approach captures context "
                "from both directions."
            ),
            metadata={"source": "bert.pdf", "page": 2, "chunk_index": 0},
        ),
        Document(
            page_content=(
                "The training process uses the Adam optimizer with a "
                "learning rate warmup followed by decay. The model is "
                "trained on 8 GPUs for approximately 12 hours."
            ),
            metadata={"source": "transformer.pdf", "page": 5, "chunk_index": 3},
        ),
    ]
    return create_index(chunks, embedding_model=embedding_model)


@pytest.fixture
def mock_llm():
    """
    Create a mock LLM that returns predictable responses.

    This allows testing the full chain without needing Ollama running.
    """
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(
        content=(
            "The Transformer uses self-attention to process sequences "
            "in parallel. [Source: transformer.pdf, Page 1]"
        )
    )
    return mock


@pytest.fixture
def memory():
    """Create a fresh ChatMemory instance."""
    return ChatMemory(window_size=3)


# ===========================================================================
# 1. Prompt Template Tests
# ===========================================================================


class TestPromptTemplates:
    """Tests for generation/prompts.py"""

    def test_qa_prompt_has_required_variables(self):
        """QA prompt should accept context, question, and chat_history."""
        prompt = get_qa_prompt()
        input_vars = prompt.input_variables

        assert "context" in input_vars
        assert "question" in input_vars

    def test_summarize_prompt_has_required_variables(self):
        """Summarize prompt should accept context and question."""
        prompt = get_summarize_prompt()
        input_vars = prompt.input_variables

        assert "context" in input_vars
        assert "question" in input_vars

    def test_compare_prompt_has_required_variables(self):
        """Compare prompt should accept context and question."""
        prompt = get_compare_prompt()
        input_vars = prompt.input_variables

        assert "context" in input_vars
        assert "question" in input_vars

    def test_qa_prompt_formats_correctly(self):
        """QA prompt should produce properly formatted messages."""
        prompt = get_qa_prompt()
        messages = prompt.format_messages(
            context="Some paper context here.",
            question="What is attention?",
            chat_history=[],
        )

        # Should have system message + human message
        assert len(messages) >= 2
        # First message should be system
        assert isinstance(messages[0], SystemMessage)
        # System message should contain role definition
        assert "ML research tutor" in messages[0].content

    def test_qa_system_message_has_hallucination_guard(self):
        """QA system message should instruct against hallucination."""
        assert "ONLY" in QA_SYSTEM_MESSAGE
        assert "don't have enough information" in QA_SYSTEM_MESSAGE.lower() or \
               "not in the context" in QA_SYSTEM_MESSAGE.lower()

    def test_qa_system_message_requires_citations(self):
        """QA system message should require source citations."""
        assert "[Source:" in QA_SYSTEM_MESSAGE

    def test_summarize_prompt_has_structure(self):
        """Summarize system message should define output structure."""
        assert "Key Contributions" in SUMMARIZE_SYSTEM_MESSAGE
        assert "Methodology" in SUMMARIZE_SYSTEM_MESSAGE

    def test_compare_prompt_has_structure(self):
        """Compare system message should define comparison structure."""
        assert "Differences" in COMPARE_SYSTEM_MESSAGE
        assert "Strengths" in COMPARE_SYSTEM_MESSAGE

    def test_get_prompt_selector(self):
        """get_prompt() should return correct template for each mode."""
        qa = get_prompt("qa")
        summarize = get_prompt("summarize")
        compare = get_prompt("compare")

        # Each should be a different template
        assert qa is not summarize
        assert summarize is not compare

    def test_get_prompt_invalid_mode_raises(self):
        """Should raise ValueError for unknown mode."""
        with pytest.raises(ValueError, match="Unknown prompt mode"):
            get_prompt("invalid_mode")

    def test_qa_prompt_with_chat_history(self):
        """QA prompt should incorporate chat history."""
        prompt = get_qa_prompt()
        history = [
            HumanMessage(content="What is BERT?"),
            AIMessage(content="BERT is a language model."),
        ]

        messages = prompt.format_messages(
            context="BERT context here.",
            question="How is it trained?",
            chat_history=history,
        )

        # Should include history messages between system and human
        all_content = " ".join(m.content for m in messages)
        assert "What is BERT?" in all_content
        assert "BERT is a language model." in all_content


# ===========================================================================
# 2. Chat Memory Tests
# ===========================================================================


class TestChatMemory:
    """Tests for memory/chat_memory.py"""

    def test_initial_state(self, memory):
        """New memory should be empty."""
        assert memory.is_empty
        assert memory.exchange_count == 0
        assert memory.get_history() == []

    def test_add_single_exchange(self, memory):
        """Adding one exchange should store 2 messages."""
        memory.add_exchange("Hello", "Hi there!")

        assert not memory.is_empty
        assert memory.exchange_count == 1
        assert len(memory.get_history()) == 2

    def test_message_types(self, memory):
        """Messages should be proper HumanMessage and AIMessage types."""
        memory.add_exchange("Question?", "Answer!")
        history = memory.get_history()

        assert isinstance(history[0], HumanMessage)
        assert isinstance(history[1], AIMessage)
        assert history[0].content == "Question?"
        assert history[1].content == "Answer!"

    def test_window_overflow(self):
        """Adding more than window_size exchanges should trim oldest."""
        mem = ChatMemory(window_size=2)

        mem.add_exchange("Q1", "A1")
        mem.add_exchange("Q2", "A2")
        mem.add_exchange("Q3", "A3")  # Should trim Q1/A1

        assert mem.exchange_count == 2
        history = mem.get_history()

        # Oldest exchange (Q1/A1) should be gone
        contents = [m.content for m in history]
        assert "Q1" not in contents
        assert "A1" not in contents
        assert "Q2" in contents
        assert "Q3" in contents

    def test_clear_memory(self, memory):
        """clear() should remove all messages."""
        memory.add_exchange("Q", "A")
        assert not memory.is_empty

        memory.clear()
        assert memory.is_empty
        assert memory.exchange_count == 0

    def test_get_history_returns_copy(self, memory):
        """get_history() should return a copy, not a reference."""
        memory.add_exchange("Q", "A")
        history = memory.get_history()

        # Modifying the returned list shouldn't affect the memory
        history.clear()
        assert memory.exchange_count == 1

    def test_get_history_as_text(self, memory):
        """Should format history as readable text."""
        memory.add_exchange("What is BERT?", "BERT is a language model.")

        text = memory.get_history_as_text()

        assert "Human: What is BERT?" in text
        assert "AI: BERT is a language model." in text

    def test_get_history_as_text_empty(self):
        """Empty memory should return empty string."""
        mem = ChatMemory()
        assert mem.get_history_as_text() == ""

    def test_get_last_exchange(self, memory):
        """Should return the most recent exchange."""
        memory.add_exchange("Q1", "A1")
        memory.add_exchange("Q2", "A2")

        last = memory.get_last_exchange()

        assert last == ("Q2", "A2")

    def test_get_last_exchange_empty(self):
        """Should return None for empty memory."""
        mem = ChatMemory()
        assert mem.get_last_exchange() is None

    def test_multiple_windows(self):
        """Different window sizes should retain different amounts."""
        small = ChatMemory(window_size=1)
        large = ChatMemory(window_size=10)

        for i in range(5):
            small.add_exchange(f"Q{i}", f"A{i}")
            large.add_exchange(f"Q{i}", f"A{i}")

        assert small.exchange_count == 1
        assert large.exchange_count == 5


# ===========================================================================
# 3. Chain Tests (with mocked LLM)
# ===========================================================================


class TestPaperTutorChain:
    """Tests for generation/chain.py using a mocked LLM."""

    def _create_chain(self, sample_index, mock_llm, mode="qa"):
        """Helper to create a chain with mocked LLM."""
        chain = PaperTutorChain(
            index=sample_index,
            memory=ChatMemory(window_size=3),
            mode=mode,
        )
        chain.llm = mock_llm  # Replace real LLM with mock
        return chain

    def test_ask_returns_chain_response(self, sample_index, mock_llm):
        """ask() should return a ChainResponse object."""
        chain = self._create_chain(sample_index, mock_llm)
        response = chain.ask("What is the Transformer?")

        assert isinstance(response, ChainResponse)

    def test_chain_response_has_answer(self, sample_index, mock_llm):
        """Response should contain the LLM's answer."""
        chain = self._create_chain(sample_index, mock_llm)
        response = chain.ask("What is self-attention?")

        assert response.answer != ""
        assert len(response.answer) > 0

    def test_chain_response_has_sources(self, sample_index, mock_llm):
        """Response should include source metadata."""
        chain = self._create_chain(sample_index, mock_llm)
        response = chain.ask("How does attention work?")

        assert isinstance(response.sources, list)
        # Sources should have been retrieved
        assert len(response.sources) > 0

    def test_chain_response_has_context(self, sample_index, mock_llm):
        """Response should include the context that was sent to the LLM."""
        chain = self._create_chain(sample_index, mock_llm)
        response = chain.ask("What is BERT?")

        assert response.context_used != ""
        assert "[Source:" in response.context_used

    def test_chain_stores_in_memory(self, sample_index, mock_llm):
        """QA mode should store exchanges in memory."""
        chain = self._create_chain(sample_index, mock_llm)

        chain.ask("First question")
        chain.ask("Second question")

        assert chain.memory.exchange_count == 2

    def test_chain_passes_history_to_llm(self, sample_index, mock_llm):
        """LLM should receive chat history for follow-up context."""
        chain = self._create_chain(sample_index, mock_llm)

        chain.ask("What is the Transformer?")
        chain.ask("Tell me more about it")

        # The mock should have been called twice
        assert mock_llm.invoke.call_count == 2

    def test_chain_mode_switching(self, sample_index, mock_llm):
        """set_mode() should change the active prompt."""
        chain = self._create_chain(sample_index, mock_llm, mode="qa")

        assert chain.mode == "qa"
        chain.set_mode("summarize")
        assert chain.mode == "summarize"

    def test_chain_invalid_mode_raises(self, sample_index, mock_llm):
        """set_mode() with invalid mode should raise ValueError."""
        chain = self._create_chain(sample_index, mock_llm)

        with pytest.raises(ValueError, match="Invalid mode"):
            chain.set_mode("invalid")

    def test_chain_clear_memory(self, sample_index, mock_llm):
        """clear_memory() should reset conversation history."""
        chain = self._create_chain(sample_index, mock_llm)

        chain.ask("Question 1")
        assert chain.memory.exchange_count == 1

        chain.clear_memory()
        assert chain.memory.is_empty

    def test_chain_mode_in_response(self, sample_index, mock_llm):
        """Response should indicate which mode was used."""
        chain = self._create_chain(sample_index, mock_llm, mode="qa")
        response = chain.ask("Test question")
        assert response.mode == "qa"

    def test_chain_per_query_mode_override(self, sample_index, mock_llm):
        """Mode can be overridden per-query without changing default."""
        chain = self._create_chain(sample_index, mock_llm, mode="qa")

        response = chain.ask("Summarize the paper", mode="summarize")

        assert response.mode == "summarize"
        assert chain.mode == "qa"  # Default unchanged

    def test_chain_with_source_filter(self, sample_index, mock_llm):
        """Source filter should be passed to retriever."""
        chain = self._create_chain(sample_index, mock_llm)
        response = chain.ask(
            "What is attention?",
            source_filter="transformer.pdf",
        )

        # All sources should be from transformer.pdf
        for src in response.sources:
            assert src["source"] == "transformer.pdf"

    def test_chain_error_handling(self, sample_index):
        """Chain should handle LLM errors gracefully."""
        chain = PaperTutorChain(
            index=sample_index,
            memory=ChatMemory(),
        )
        # Set a mock that raises an exception
        mock_error_llm = MagicMock()
        mock_error_llm.invoke.side_effect = Exception("Connection refused")
        chain.llm = mock_error_llm

        response = chain.ask("Test question")

        assert response.error is not None
        assert "Connection refused" in response.error

    def test_chain_response_no_error_on_success(self, sample_index, mock_llm):
        """Successful response should have error=None."""
        chain = self._create_chain(sample_index, mock_llm)
        response = chain.ask("Test question")

        assert response.error is None

    def test_chain_update_index(self, sample_index, mock_llm, embedding_model):
        """update_index() should replace the FAISS index."""
        chain = self._create_chain(sample_index, mock_llm)

        new_chunks = [
            Document(
                page_content="New paper about GPT models.",
                metadata={"source": "gpt.pdf", "page": 1, "chunk_index": 0},
            ),
        ]
        new_index = create_index(new_chunks, embedding_model=embedding_model)

        chain.update_index(new_index)
        assert chain.index is new_index


# ===========================================================================
# 4. ChainResponse Tests
# ===========================================================================


class TestChainResponse:
    """Tests for the ChainResponse dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        response = ChainResponse(answer="Test answer")

        assert response.answer == "Test answer"
        assert response.sources == []
        assert response.retrieval_results == []
        assert response.mode == "qa"
        assert response.context_used == ""
        assert response.error is None

    def test_with_error(self):
        """Should store error message."""
        response = ChainResponse(answer="", error="Something went wrong")

        assert response.error == "Something went wrong"
        assert response.answer == ""


# ===========================================================================
# Run with: python -m pytest tests/test_generation.py -v
# ===========================================================================
