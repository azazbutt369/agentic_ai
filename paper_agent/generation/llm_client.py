"""
==============================================================================
LLM Client — Local inference via Ollama
==============================================================================

WHY OLLAMA:
    Ollama makes running LLMs locally dead simple:
    - One command to pull a model: `ollama pull qwen2.5:1.5b`
    - Runs an OpenAI-compatible HTTP server on localhost:11434
    - Handles model loading, quantization, and memory management
    - Cross-platform (Windows, macOS, Linux)
    - No GPU required (but uses it if available)

WHY ChatOllama (not Ollama):
    LangChain provides two Ollama wrappers:
    - `Ollama` (legacy): Completion-style, no chat formatting
    - `ChatOllama`: Chat-style with system/human/AI message roles
    We use ChatOllama because instruction-tuned models like Qwen2.5
    perform MUCH better with proper chat formatting.

DESIGN DECISIONS:
    - Temperature 0.3: Low enough to reduce hallucination, high enough
      to avoid robotic repetition.
    - num_ctx=2048: Conservative context window to keep RAM usage low.
      Qwen2.5-1.5B supports up to 32k, but we only need ~2k for our
      chunk sizes and chat history.
    - The client is a singleton (cached) to avoid reconnection overhead.
==============================================================================
"""

import logging
from functools import lru_cache

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from config.settings import (
    OLLAMA_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_NUM_CTX,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    num_ctx: int | None = None,
    base_url: str | None = None,
) -> ChatOllama:
    """
    Create and cache an Ollama LLM client.

    The client connects to the local Ollama server and sends
    inference requests via HTTP. The model must already be pulled
    (`ollama pull <model>`).

    Args:
        model: Ollama model identifier (e.g., "qwen2.5:1.5b").
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
        num_ctx: Context window size in tokens.
        base_url: Ollama server URL.

    Returns:
        Configured ChatOllama instance.
    """
    m = model or LLM_MODEL
    t = temperature if temperature is not None else LLM_TEMPERATURE
    ctx = num_ctx or LLM_NUM_CTX
    url = base_url or OLLAMA_BASE_URL

    logger.info(
        f"Connecting to Ollama: model={m}, temp={t}, "
        f"ctx={ctx}, url={url}"
    )

    llm = ChatOllama(
        model=m,
        temperature=t,
        num_ctx=ctx,
        base_url=url,
    )

    logger.info(f"LLM client ready: {m}")
    return llm


def check_ollama_connection(base_url: str | None = None) -> dict:
    """
    Verify that the Ollama server is running and the model is available.

    Useful for the Streamlit UI to show connection status.

    Returns:
        Dict with: connected (bool), model (str), error (str | None)
    """
    url = base_url or OLLAMA_BASE_URL

    try:
        import requests
        response = requests.get(f"{url}/api/tags", timeout=5)
        response.raise_for_status()

        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]

        target = LLM_MODEL
        available = any(target in name for name in model_names)

        if available:
            return {
                "connected": True,
                "model": target,
                "error": None,
                "available_models": model_names,
            }
        else:
            return {
                "connected": True,
                "model": target,
                "error": (
                    f"Model '{target}' not found. "
                    f"Available: {model_names}. "
                    f"Run: ollama pull {target}"
                ),
                "available_models": model_names,
            }

    except Exception as e:
        return {
            "connected": False,
            "model": LLM_MODEL,
            "error": (
                f"Cannot connect to Ollama at {url}. "
                f"Is Ollama running? Start with: ollama serve\n"
                f"Error: {e}"
            ),
            "available_models": [],
        }
