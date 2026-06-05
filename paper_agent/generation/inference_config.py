"""
==============================================================================
Inference Configuration — Settings for remote GPU inference
==============================================================================

WHY A SEPARATE CONFIG:
    Remote inference has its own set of parameters (URL, timeout, retries,
    cache size) that are independent of the local Ollama config. Keeping
    them separate ensures the local pipeline stays untouched.

    All settings are overridable via .env for easy switching between
    local and remote modes.
==============================================================================
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Remote Inference Server
# ---------------------------------------------------------------------------

# Whether to use remote GPU inference instead of local Ollama.
# When False, the system works exactly as before via local Ollama.
REMOTE_INFERENCE_ENABLED: bool = os.getenv(
    "REMOTE_INFERENCE_ENABLED", "false"
).lower() == "true"

# The ngrok public URL for the Colab FastAPI server.
# Example: "https://abc123.ngrok-free.app"
REMOTE_INFERENCE_URL: str = os.getenv("REMOTE_INFERENCE_URL", "")

# ---------------------------------------------------------------------------
# Timeouts & Retry
# ---------------------------------------------------------------------------

# Maximum time (seconds) to wait for the full response.
# GPU inference is fast (~2-5s) but network adds latency.
REMOTE_TIMEOUT: int = int(os.getenv("REMOTE_TIMEOUT", "60"))

# Maximum time (seconds) to wait for the connection to establish.
REMOTE_CONNECT_TIMEOUT: int = int(os.getenv("REMOTE_CONNECT_TIMEOUT", "10"))

# Number of retry attempts on transient failures (network errors, 5xx).
REMOTE_MAX_RETRIES: int = int(os.getenv("REMOTE_MAX_RETRIES", "3"))

# Base delay between retries (seconds). Uses exponential backoff: 1s, 2s, 4s.
REMOTE_RETRY_DELAY: float = float(os.getenv("REMOTE_RETRY_DELAY", "1.0"))

# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

# Whether to stream tokens from the remote server via SSE.
# When True, the UI shows tokens as they arrive (lower perceived latency).
REMOTE_STREAMING: bool = os.getenv(
    "REMOTE_STREAMING", "true"
).lower() == "true"

# ---------------------------------------------------------------------------
# Response Caching
# ---------------------------------------------------------------------------

# Maximum number of cached responses (LRU eviction).
# Caching avoids re-running inference for identical queries.
RESPONSE_CACHE_SIZE: int = int(os.getenv("RESPONSE_CACHE_SIZE", "64"))

# Whether response caching is enabled.
RESPONSE_CACHE_ENABLED: bool = os.getenv(
    "RESPONSE_CACHE_ENABLED", "true"
).lower() == "true"

# ---------------------------------------------------------------------------
# Generation Parameters (sent to the remote server)
# ---------------------------------------------------------------------------

# Maximum tokens to generate. Lower = faster response.
REMOTE_MAX_TOKENS: int = int(os.getenv("REMOTE_MAX_TOKENS", "512"))

# Temperature for remote generation.
REMOTE_TEMPERATURE: float = float(os.getenv("REMOTE_TEMPERATURE", "0.3"))

# Repetition penalty to avoid loops in small models.
REMOTE_REPETITION_PENALTY: float = float(
    os.getenv("REMOTE_REPETITION_PENALTY", "1.1")
)

# Top-p (nucleus sampling). 0.9 = consider tokens covering 90% probability.
REMOTE_TOP_P: float = float(os.getenv("REMOTE_TOP_P", "0.9"))
