"""
==============================================================================
Chat Memory — Conversation history management
==============================================================================

WHY CONVERSATION MEMORY:
    Without memory, every query is independent — the user can't ask follow-up
    questions like "Can you explain that in more detail?" because the model
    has no idea what "that" refers to.

    Memory stores the last N exchanges (human + AI messages) and injects
    them into the prompt so the model can:
        - Resolve pronouns ("it", "this", "that method")
        - Avoid repeating information already discussed
        - Build on previous explanations progressively

WHY BufferWindowMemory (k=6):
    We store the last 6 exchanges (12 messages total). This is a trade-off:

    - WHY NOT unlimited: With a 2048-token context window, unlimited history
      would crowd out retrieved chunks. 6 exchanges ≈ 400-600 tokens, leaving
      room for system prompt (~200 tokens) + 4 chunks (~1200 tokens).

    - WHY NOT k=2: Too short — the model forgets context quickly. 6 exchanges
      covers a typical "drill-down" conversation:
        1. "What is this paper about?"
        2. "How does the attention mechanism work?"
        3. "What about multi-head attention?"
        4. "How does it compare to recurrent models?"
        5. "What are the limitations?"
        6. "Can you summarize the key takeaways?"

DESIGN DECISIONS:
    - We use LangChain's ChatMessageHistory + manual window management
      for simplicity and control.
    - Memory is SESSION-scoped — each Streamlit session gets its own
      memory instance. This is handled by st.session_state in the UI.
    - Messages are stored as LangChain message objects (HumanMessage,
      AIMessage) for proper chat formatting.
==============================================================================
"""

import logging
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from config.settings import MEMORY_WINDOW_SIZE

logger = logging.getLogger(__name__)


class ChatMemory:
    """
    Sliding window conversation memory.

    Stores the last `window_size` exchanges (human + AI pairs) and
    provides them as a list of messages for prompt injection.

    Attributes:
        window_size: Maximum number of exchanges to retain.
        messages: List of all messages (HumanMessage and AIMessage).
    """

    def __init__(self, window_size: int | None = None):
        """
        Initialize chat memory.

        Args:
            window_size: Number of recent exchanges to keep.
                         Default from settings (6).
        """
        self.window_size = window_size or MEMORY_WINDOW_SIZE
        self.messages: list[BaseMessage] = []

        logger.debug(f"ChatMemory initialized with window_size={self.window_size}")

    def add_exchange(self, human_message: str, ai_message: str) -> None:
        """
        Add a human-AI exchange to memory.

        Each exchange consists of a human message (the user's question)
        and an AI message (the agent's response). They are always added
        as a pair to maintain conversation coherence.

        Args:
            human_message: The user's question/input.
            ai_message: The agent's response.
        """
        self.messages.append(HumanMessage(content=human_message))
        self.messages.append(AIMessage(content=ai_message))

        # Trim to window size (each exchange = 2 messages)
        max_messages = self.window_size * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

        logger.debug(
            f"Memory: {len(self.messages)} messages "
            f"({len(self.messages) // 2} exchanges)"
        )

    def get_history(self) -> list[BaseMessage]:
        """
        Get the current conversation history as a list of messages.

        Returns:
            List of HumanMessage and AIMessage objects, in chronological order.
            This is directly compatible with LangChain's MessagesPlaceholder.
        """
        return self.messages.copy()

    def get_history_as_text(self) -> str:
        """
        Get conversation history as a formatted text string.

        Useful for debugging or when the prompt template expects
        a string instead of message objects.

        Returns:
            Formatted conversation history string.
        """
        if not self.messages:
            return ""

        lines = []
        for msg in self.messages:
            role = "Human" if isinstance(msg, HumanMessage) else "AI"
            lines.append(f"{role}: {msg.content}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all conversation history."""
        self.messages.clear()
        logger.debug("Chat memory cleared")

    @property
    def exchange_count(self) -> int:
        """Number of exchanges currently in memory."""
        return len(self.messages) // 2

    @property
    def is_empty(self) -> bool:
        """Whether the memory has any exchanges."""
        return len(self.messages) == 0

    def get_last_exchange(self) -> tuple[str, str] | None:
        """
        Get the most recent exchange (human + AI).

        Returns:
            Tuple of (human_message, ai_message) or None if empty.
        """
        if len(self.messages) < 2:
            return None

        human = self.messages[-2]
        ai = self.messages[-1]

        return (human.content, ai.content)
