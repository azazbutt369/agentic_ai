"""
==============================================================================
Streaming Utilities — Parse SSE streams and adapt for Streamlit
==============================================================================

WHY STREAMING:
    Without streaming, the user stares at a spinner for 5-10 seconds while
    the full response generates. With streaming, tokens appear as they're
    generated — the first token shows up in <1 second, dramatically
    reducing PERCEIVED latency even if total generation time is the same.

DESIGN:
    - parse_sse_stream(): Parses Server-Sent Events from the FastAPI server
    - StreamingAdapter: Wraps the SSE stream into a generator that Streamlit's
      st.write_stream() can consume directly.
==============================================================================
"""

import json
import logging
from typing import Generator, Iterator

logger = logging.getLogger(__name__)


def parse_sse_stream(response_iter: Iterator[bytes]) -> Generator[str, None, None]:
    """
    Parse a Server-Sent Events (SSE) byte stream into text tokens.

    SSE format (from our FastAPI server):
        data: {"token": "Hello"}
        data: {"token": " world"}
        data: [DONE]

    Args:
        response_iter: Iterator of raw bytes from httpx streaming response.

    Yields:
        Individual text tokens as strings.
    """
    buffer = ""

    for chunk in response_iter:
        # Decode bytes to string
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", errors="replace")

        buffer += chunk

        # Process complete lines
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()

            if not line:
                continue

            # SSE lines start with "data: "
            if line.startswith("data: "):
                data = line[6:]  # Remove "data: " prefix

                # Check for stream end signal
                if data.strip() == "[DONE]":
                    return

                # Parse JSON payload
                try:
                    payload = json.loads(data)
                    token = payload.get("token", "")
                    if token:
                        yield token
                except json.JSONDecodeError:
                    # Non-JSON data line — might be a plain text token
                    if data.strip():
                        yield data


def collect_stream(token_generator: Generator[str, None, None]) -> str:
    """
    Collect all tokens from a streaming generator into a single string.

    Useful when streaming is enabled at the transport layer but the caller
    wants the complete response (e.g., for caching or non-streaming UI).

    Args:
        token_generator: Generator yielding individual tokens.

    Returns:
        Complete response as a single string.
    """
    tokens = []
    for token in token_generator:
        tokens.append(token)
    return "".join(tokens)


class StreamBuffer:
    """
    Accumulates streamed tokens and provides the full text on demand.

    Used by the chain to both stream tokens to the UI AND capture
    the complete response for memory/caching.
    """

    def __init__(self):
        self.tokens: list[str] = []
        self.complete = False

    def add_token(self, token: str) -> None:
        """Add a token to the buffer."""
        self.tokens.append(token)

    def get_text(self) -> str:
        """Get the complete accumulated text."""
        return "".join(self.tokens)

    def stream_and_capture(
        self, token_generator: Generator[str, None, None]
    ) -> Generator[str, None, None]:
        """
        Wrap a token generator to both yield tokens AND capture them.

        This lets the UI stream tokens while the chain captures the
        full response for memory storage.

        Yields:
            Individual tokens (pass-through from the generator).
        """
        for token in token_generator:
            self.add_token(token)
            yield token

        self.complete = True
