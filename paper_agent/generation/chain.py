"""
==============================================================================
RAG Chain — Orchestrate retrieval + generation into a single pipeline
==============================================================================

WHY A CHAIN:
    The chain is the glue that connects all components:
        Query → Retrieve → Format Context → Build Prompt → Generate → Parse

    Without a chain, the caller (Streamlit UI) would need to manually:
        1. Call retriever.retrieve()
        2. Call format_context()
        3. Build the prompt with context + history
        4. Call the LLM
        5. Parse the response

    The chain encapsulates all of this into a single `ask()` call.

DESIGN DECISIONS:
    - We build the chain MANUALLY instead of using LangChain's
      ConversationalRetrievalChain because:
        * More control over each step
        * Easier to debug (each step is logged)
        * Simpler to understand for educational purposes
        * ConversationalRetrievalChain is deprecated in LangChain v0.3+

    - The chain returns a structured result with the answer, sources,
      and retrieval metadata — not just a raw string.

    - Mode switching (QA/Summarize/Compare) is handled by swapping
      the prompt template, not by creating separate chains.
==============================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Generator, Optional

from langchain_community.vectorstores import FAISS

from generation.llm_client import get_llm
from generation.prompts import get_prompt
from generation.remote_inference import (
    remote_generate,
    check_remote_connection,
    get_cache,
)
from generation.streaming_utils import StreamBuffer, collect_stream
from memory.chat_memory import ChatMemory
from retrieval.retriever import (
    retrieve,
    format_context,
    get_sources_summary,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


@dataclass
class ChainResponse:
    """
    Structured response from the RAG chain.

    WHY a dataclass:
        The Streamlit UI needs more than just the answer text — it needs
        sources for citation display, scores for confidence indicators,
        and retrieval metadata for debugging.
    """
    answer: str                                     # The LLM's response text
    sources: list[dict] = field(default_factory=list)  # Source summaries for UI
    retrieval_results: list[RetrievalResult] = field(default_factory=list)
    mode: str = "qa"                                # Which prompt was used
    context_used: str = ""                          # The raw context sent to LLM
    error: str | None = None                        # Error message if any


class PaperTutorChain:
    """
    Main RAG chain that orchestrates retrieval → generation.

    This is the primary interface used by the Streamlit UI.

    Usage:
        chain = PaperTutorChain(index=faiss_index)
        response = chain.ask("How does the Transformer work?")
        print(response.answer)
        print(response.sources)
    """

    def __init__(
        self,
        index: FAISS,
        memory: ChatMemory | None = None,
        mode: str = "qa",
        use_remote: bool = False,
        remote_url: str | None = None,
    ):
        """
        Initialize the RAG chain.

        Args:
            index: FAISS vector store with embedded paper chunks.
            memory: Chat memory for conversation history. Created if None.
            mode: Default prompt mode ("qa", "summarize", "compare").
            use_remote: If True, use remote GPU server instead of local Ollama.
            remote_url: Override the remote inference URL.
        """
        self.index = index
        self.memory = memory or ChatMemory()
        self.mode = mode
        self.use_remote = use_remote
        self.remote_url = remote_url

        # Only load local LLM if not using remote
        if not use_remote:
            self.llm = get_llm()
        else:
            self.llm = None

        inference_mode = "REMOTE GPU" if use_remote else "LOCAL Ollama"
        logger.info(f"PaperTutorChain initialized in '{mode}' mode [{inference_mode}]")

    def ask(
        self,
        question: str,
        mode: str | None = None,
        k: int | None = None,
        threshold: float | None = None,
        source_filter: str | None = None,
    ) -> ChainResponse:
        """
        Process a user question through the full RAG pipeline.

        Pipeline:
            1. Retrieve relevant chunks from FAISS
            2. Format chunks into context with citations
            3. Build prompt (system + context + history + question)
            4. Generate response via Ollama
            5. Store exchange in memory
            6. Return structured response

        Args:
            question: The user's question.
            mode: Override the prompt mode for this query.
            k: Number of chunks to retrieve.
            threshold: Minimum similarity score.
            source_filter: Restrict to a specific source file.

        Returns:
            ChainResponse with answer, sources, and metadata.
        """
        active_mode = mode or self.mode

        try:
            # Step 1: Retrieve relevant chunks
            logger.info(f"Step 1/4: Retrieving chunks for: '{question[:60]}'")
            results = retrieve(
                index=self.index,
                query=question,
                k=k,
                threshold=threshold,
                source_filter=source_filter,
            )

            # Step 2: Format context
            logger.info(f"Step 2/4: Formatting {len(results)} chunks into context")
            context = format_context(results)

            # Step 3: Build and invoke prompt
            logger.info(f"Step 3/4: Generating response in '{active_mode}' mode")
            prompt = get_prompt(active_mode)

            # Build the message inputs
            prompt_inputs = {
                "context": context,
                "question": question,
            }

            # Add chat history for QA mode (other modes don't use history)
            if active_mode == "qa":
                prompt_inputs["chat_history"] = self.memory.get_history()

            messages = prompt.format_messages(**prompt_inputs)

            # Step 4: Generate response
            if self.use_remote:
                # REMOTE PATH: Send formatted prompt to Colab GPU server
                prompt_text = "\n".join(m.content for m in messages)
                answer = remote_generate(
                    prompt=prompt_text,
                    url=self.remote_url,
                    stream=False,
                )
            else:
                # LOCAL PATH: Use Ollama (existing behavior, unchanged)
                response = self.llm.invoke(messages)
                answer = response.content

            logger.info(
                f"Step 4/4: Response generated "
                f"({len(answer)} chars)"
            )

            # Store in memory (QA mode only — summaries don't need history)
            if active_mode == "qa":
                self.memory.add_exchange(question, answer)

            return ChainResponse(
                answer=answer,
                sources=get_sources_summary(results),
                retrieval_results=results,
                mode=active_mode,
                context_used=context,
            )

        except Exception as e:
            logger.error(f"Chain error: {e}")
            return ChainResponse(
                answer="",
                mode=active_mode,
                error=str(e),
            )

    def ask_stream(
        self,
        question: str,
        mode: str | None = None,
        k: int | None = None,
        threshold: float | None = None,
        source_filter: str | None = None,
    ) -> tuple[Generator[str, None, None], "StreamBuffer", list, list]:
        """
        Process a query and return a streaming token generator.

        Only works when use_remote=True. Returns a tuple of:
            (token_generator, buffer, sources, retrieval_results)

        The caller should iterate the generator to get tokens, then
        call buffer.get_text() for the complete response.
        """
        if not self.use_remote:
            raise RuntimeError("Streaming requires remote inference (use_remote=True)")

        active_mode = mode or self.mode

        # Steps 1-3: Same as ask()
        results = retrieve(
            index=self.index, query=question,
            k=k, threshold=threshold, source_filter=source_filter,
        )
        context = format_context(results)
        prompt = get_prompt(active_mode)

        prompt_inputs = {"context": context, "question": question}
        if active_mode == "qa":
            prompt_inputs["chat_history"] = self.memory.get_history()

        messages = prompt.format_messages(**prompt_inputs)
        prompt_text = "\n".join(m.content for m in messages)

        # Get streaming generator from remote server
        token_gen = remote_generate(
            prompt=prompt_text,
            url=self.remote_url,
            stream=True,
        )

        # Wrap with buffer to capture the full response
        buffer = StreamBuffer()
        wrapped_gen = buffer.stream_and_capture(token_gen)

        sources = get_sources_summary(results)
        return wrapped_gen, buffer, sources, results

    def set_mode(self, mode: str) -> None:
        """
        Switch the default prompt mode.

        Args:
            mode: "qa", "summarize", or "compare".
        """
        if mode not in ("qa", "summarize", "compare"):
            raise ValueError(f"Invalid mode: {mode}")

        self.mode = mode
        logger.info(f"Chain mode switched to '{mode}'")

    def clear_memory(self) -> None:
        """Clear conversation history."""
        self.memory.clear()
        logger.info("Chain memory cleared")

    def update_index(self, index: FAISS) -> None:
        """
        Update the FAISS index (e.g., after ingesting a new paper).

        Args:
            index: New or updated FAISS index.
        """
        self.index = index
        logger.info("Chain index updated")
