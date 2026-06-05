"""
==============================================================================
Remote Inference Client — Async HTTP client for Colab GPU server
==============================================================================

WHY THIS EXISTS:
    This module replaces `self.llm.invoke(messages)` in the chain when
    remote GPU inference is enabled. Instead of calling local Ollama,
    it sends the formatted prompt text to the FastAPI server running
    on Google Colab via HTTP POST.

WHAT IT DOES:
    1. Sends the pre-formatted prompt to the remote server
    2. Supports both streaming (SSE) and non-streaming responses
    3. Implements retry with exponential backoff for network resilience
    4. Uses response caching to skip inference for repeated queries
    5. Handles timeouts, connection errors, and HTTP errors gracefully

DESIGN DECISIONS:
    - Uses `httpx` (not `requests`) for proper async and streaming support
    - The prompt is sent as a PLAIN TEXT string, not as LangChain message
      objects — the Colab server doesn't need LangChain installed
    - Retry logic uses exponential backoff: 1s → 2s → 4s
    - The client is stateless — each call is independent. Connection
      pooling is handled by httpx internally.
==============================================================================
"""

import time
import logging
from typing import Generator, Optional

import httpx

from generation.inference_config import (
    REMOTE_INFERENCE_URL,
    REMOTE_TIMEOUT,
    REMOTE_CONNECT_TIMEOUT,
    REMOTE_MAX_RETRIES,
    REMOTE_RETRY_DELAY,
    REMOTE_STREAMING,
    REMOTE_MAX_TOKENS,
    REMOTE_TEMPERATURE,
    REMOTE_REPETITION_PENALTY,
    REMOTE_TOP_P,
)
from generation.response_cache import ResponseCache
from generation.streaming_utils import parse_sse_stream, collect_stream

logger = logging.getLogger(__name__)

# Module-level cache instance (shared across calls)
_response_cache = ResponseCache()


def get_cache() -> ResponseCache:
    """Get the module-level response cache instance."""
    return _response_cache


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def check_remote_connection(url: str | None = None) -> dict:
    """
    Check if the remote Colab inference server is reachable.

    Args:
        url: Override the configured remote URL.

    Returns:
        Dict with: connected (bool), url (str), error (str | None),
                    model (str | None), latency_ms (float).
    """
    target_url = url or REMOTE_INFERENCE_URL

    if not target_url:
        return {
            "connected": False,
            "url": "",
            "error": "Remote inference URL not configured",
            "model": None,
            "latency_ms": 0,
        }

    try:
        start = time.perf_counter()
        resp = httpx.get(
            f"{target_url.rstrip('/')}/health",
            timeout=5.0,
            follow_redirects=True,
        )
        latency = (time.perf_counter() - start) * 1000
        resp.raise_for_status()

        data = resp.json()
        return {
            "connected": True,
            "url": target_url,
            "error": None,
            "model": data.get("model", "unknown"),
            "latency_ms": round(latency, 1),
        }

    except Exception as e:
        return {
            "connected": False,
            "url": target_url,
            "error": str(e),
            "model": None,
            "latency_ms": 0,
        }


# ---------------------------------------------------------------------------
# Non-streaming inference
# ---------------------------------------------------------------------------


def generate_remote(
    prompt: str,
    url: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    use_cache: bool = True,
) -> str:
    """
    Send a prompt to the remote GPU server and get the full response.

    This is the NON-STREAMING path — waits for the complete response
    before returning. Simpler but higher perceived latency.

    Args:
        prompt: The fully formatted prompt text (system + context + question).
        url: Override the configured remote URL.
        max_tokens: Override max generation tokens.
        temperature: Override generation temperature.
        use_cache: Whether to check/store in response cache.

    Returns:
        The generated response text.

    Raises:
        ConnectionError: If the server is unreachable after retries.
        RuntimeError: If the server returns a non-200 status.
    """
    target_url = url or REMOTE_INFERENCE_URL

    if not target_url:
        raise ConnectionError(
            "Remote inference URL not set. "
            "Set REMOTE_INFERENCE_URL in .env or paste the ngrok URL in the UI."
        )

    # Check cache first
    if use_cache:
        cached = _response_cache.get(prompt)
        if cached is not None:
            logger.info("Using cached response (skipping remote inference)")
            return cached

    # Build request payload
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens or REMOTE_MAX_TOKENS,
        "temperature": temperature if temperature is not None else REMOTE_TEMPERATURE,
        "repetition_penalty": REMOTE_REPETITION_PENALTY,
        "top_p": REMOTE_TOP_P,
        "stream": False,
    }

    # Retry loop with exponential backoff
    last_error = None
    for attempt in range(REMOTE_MAX_RETRIES):
        try:
            logger.info(
                f"Remote inference attempt {attempt + 1}/{REMOTE_MAX_RETRIES} "
                f"→ {target_url}"
            )

            resp = httpx.post(
                f"{target_url.rstrip('/')}/generate",
                json=payload,
                timeout=httpx.Timeout(
                    connect=REMOTE_CONNECT_TIMEOUT,
                    read=REMOTE_TIMEOUT,
                    write=10.0,
                    pool=10.0,
                ),
                follow_redirects=True,
            )
            resp.raise_for_status()

            data = resp.json()
            answer = data.get("response", "")

            # Cache the response
            if use_cache and answer:
                _response_cache.put(prompt, answer)

            logger.info(
                f"Remote inference complete "
                f"({len(answer)} chars, {data.get('generation_time_ms', '?')}ms)"
            )
            return answer

        except httpx.TimeoutException as e:
            last_error = f"Timeout after {REMOTE_TIMEOUT}s: {e}"
            logger.warning(f"Attempt {attempt + 1} timeout: {e}")

        except httpx.HTTPStatusError as e:
            last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            logger.warning(f"Attempt {attempt + 1} HTTP error: {last_error}")

            # Don't retry 4xx errors (client-side problem)
            if 400 <= e.response.status_code < 500:
                raise RuntimeError(last_error)

        except httpx.ConnectError as e:
            last_error = f"Connection failed: {e}"
            logger.warning(f"Attempt {attempt + 1} connection error: {e}")

        except Exception as e:
            last_error = f"Unexpected error: {e}"
            logger.warning(f"Attempt {attempt + 1} error: {e}")

        # Exponential backoff before retry
        if attempt < REMOTE_MAX_RETRIES - 1:
            delay = REMOTE_RETRY_DELAY * (2 ** attempt)
            logger.info(f"Retrying in {delay}s...")
            time.sleep(delay)

    raise ConnectionError(
        f"Remote inference failed after {REMOTE_MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Streaming inference
# ---------------------------------------------------------------------------


def generate_remote_stream(
    prompt: str,
    url: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> Generator[str, None, None]:
    """
    Send a prompt to the remote GPU server and stream tokens back.

    This is the STREAMING path — tokens are yielded as they're generated.
    First token typically arrives in <1 second.

    Args:
        prompt: The fully formatted prompt text.
        url: Override the configured remote URL.
        max_tokens: Override max generation tokens.
        temperature: Override generation temperature.

    Yields:
        Individual text tokens as they're generated.

    Raises:
        ConnectionError: If the server is unreachable.
    """
    target_url = url or REMOTE_INFERENCE_URL

    if not target_url:
        raise ConnectionError("Remote inference URL not set.")

    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens or REMOTE_MAX_TOKENS,
        "temperature": temperature if temperature is not None else REMOTE_TEMPERATURE,
        "repetition_penalty": REMOTE_REPETITION_PENALTY,
        "top_p": REMOTE_TOP_P,
        "stream": True,
    }

    try:
        logger.info(f"Starting streaming inference → {target_url}")

        with httpx.stream(
            "POST",
            f"{target_url.rstrip('/')}/generate",
            json=payload,
            timeout=httpx.Timeout(
                connect=REMOTE_CONNECT_TIMEOUT,
                read=REMOTE_TIMEOUT,
                write=10.0,
                pool=10.0,
            ),
            follow_redirects=True,
        ) as response:
            response.raise_for_status()
            yield from parse_sse_stream(response.iter_lines())

    except httpx.TimeoutException as e:
        raise ConnectionError(f"Streaming timeout: {e}")
    except httpx.ConnectError as e:
        raise ConnectionError(f"Cannot connect to {target_url}: {e}")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Server error {e.response.status_code}: "
            f"{e.response.text[:200]}"
        )


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------


def remote_generate(
    prompt: str,
    url: str | None = None,
    stream: bool | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    use_cache: bool = True,
) -> str | Generator[str, None, None]:
    """
    Unified remote inference interface — auto-selects streaming or non-streaming.

    When streaming, returns a Generator that yields tokens.
    When non-streaming, returns the complete response string.

    Args:
        prompt: The fully formatted prompt text.
        url: Override the configured remote URL.
        stream: Override streaming preference.
        max_tokens: Override max generation tokens.
        temperature: Override generation temperature.
        use_cache: Whether to use response cache (non-streaming only).

    Returns:
        str (non-streaming) or Generator[str, None, None] (streaming).
    """
    do_stream = stream if stream is not None else REMOTE_STREAMING

    if do_stream:
        return generate_remote_stream(
            prompt=prompt,
            url=url,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        return generate_remote(
            prompt=prompt,
            url=url,
            max_tokens=max_tokens,
            temperature=temperature,
            use_cache=use_cache,
        )
