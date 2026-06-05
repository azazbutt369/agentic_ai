"""
==============================================================================
ML Paper Tutor Agent — Central Configuration
==============================================================================

WHY THIS EXISTS:
    Every parameter that controls the agent's behavior lives here in one place.
    This avoids magic numbers scattered across modules and makes it trivial to
    tune the system (e.g., switch models, change chunk sizes) without hunting
    through code.

DESIGN DECISIONS:
    - All values have sensible defaults that work on low-resource machines.
    - Environment variables (via .env) override defaults for flexibility.
    - Paths are resolved relative to the project root so the system works
      regardless of where it's invoked from.
==============================================================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env file if present (no error if missing — defaults kick in)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
# Resolve the project root as two levels up from this file:
#   config/settings.py -> config/ -> paper_agent/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where uploaded PDFs are stored
PAPERS_DIR = PROJECT_ROOT / os.getenv("PAPERS_PATH", "data/papers")

# Where the FAISS index is persisted to disk
FAISS_INDEX_DIR = PROJECT_ROOT / os.getenv("FAISS_INDEX_PATH", "data/faiss_index")

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------
# Ollama server URL — default is localhost on Ollama's default port
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Model identifier as registered in Ollama
# WHY Qwen2.5-1.5B: Best quality-to-size ratio for instruction following.
#   - Runs comfortably on 4 GB RAM
#   - Strong at structured QA tasks
#   - Instruction-tuned (follows system prompts reliably)
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen2.5:1.5b")

# Temperature controls randomness. 0.3 = mostly deterministic but not robotic.
# WHY 0.3: Higher values (0.7+) cause a 1.5B model to hallucinate heavily.
#           Lower values (0.0) make it repetitive. 0.3 is the sweet spot.
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# Context window size in tokens. Qwen2.5-1.5B supports up to 32k, but we
# use 2048 to keep memory usage low and inference fast on CPU/low VRAM.
LLM_NUM_CTX: int = int(os.getenv("LLM_NUM_CTX", "2048"))

# ---------------------------------------------------------------------------
# Embedding Configuration
# ---------------------------------------------------------------------------
# WHY bge-small-en-v1.5:
#   - 384-dimensional embeddings (compact → fast FAISS search)
#   - ~130 MB model size (loads quickly, fits in RAM)
#   - Top-tier on MTEB benchmark for its size class
#   - Trained specifically for retrieval tasks
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Embedding dimension — must match the model's output dimension.
# Used for FAISS index initialization and validation.
EMBEDDING_DIMENSION: int = 384

# ---------------------------------------------------------------------------
# Chunking Configuration
# ---------------------------------------------------------------------------
# WHY 512 tokens:
#   - Small enough that 4 chunks + prompt + chat history fit in 2048 context
#   - Large enough to capture a full paragraph / idea
#   - Academic papers have dense, self-contained paragraphs ~300-600 tokens
#
# WHY 64-token overlap:
#   - ~12% overlap ensures no information is lost at chunk boundaries
#   - Prevents splitting a key sentence between two chunks
#   - Low enough to avoid excessive duplication in the index
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

# Separators for RecursiveCharacterTextSplitter — tried in order.
# WHY this order:
#   1. "\n\n" → paragraph breaks (highest semantic boundary)
#   2. "\n"   → line breaks (section-level)
#   3. ". "   → sentence boundaries
#   4. " "    → word boundaries (last resort)
CHUNK_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " "]

# ---------------------------------------------------------------------------
# Retrieval Configuration
# ---------------------------------------------------------------------------
# Number of chunks to retrieve per query
# WHY 4: With 512-token chunks, 4 chunks ≈ 2048 tokens of context.
#         After system prompt (~200 tokens) + chat history (~400 tokens),
#         we still fit comfortably in the 2048 context window.
TOP_K: int = int(os.getenv("TOP_K", "4"))

# Minimum cosine similarity score to include a chunk in results.
# WHY 0.35: Below this, chunks are usually noise / unrelated sections.
#           Empirically tuned — raise to 0.5 for stricter retrieval.
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

# Whether to use MMR (Maximal Marginal Relevance) for diverse retrieval.
# WHY MMR: Pure similarity search often returns redundant chunks from the
#          same paragraph. MMR balances relevance with diversity — critical
#          for "compare methods" queries that need chunks from different sections.
USE_MMR: bool = os.getenv("USE_MMR", "true").lower() == "true"

# MMR diversity weight (0 = pure similarity, 1 = pure diversity)
MMR_LAMBDA: float = float(os.getenv("MMR_LAMBDA", "0.5"))

# ---------------------------------------------------------------------------
# Conversation Memory Configuration
# ---------------------------------------------------------------------------
# Number of recent exchanges (human + AI pairs) to keep in memory.
# WHY 6: 6 exchanges ≈ 400-600 tokens — fits in context window alongside
#         retrieved chunks. More than 6 starts crowding out retrieval context.
MEMORY_WINDOW_SIZE: int = int(os.getenv("MEMORY_WINDOW_SIZE", "6"))

# ---------------------------------------------------------------------------
# Utility: Ensure data directories exist
# ---------------------------------------------------------------------------
def ensure_directories() -> None:
    """Create required data directories if they don't exist."""
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Remote GPU Inference (ADDITIVE — does not affect local Ollama pipeline)
# ---------------------------------------------------------------------------
# These settings control the optional Colab GPU inference offloading.
# When REMOTE_INFERENCE_ENABLED is False (default), the system works
# exactly as before via local Ollama. See generation/inference_config.py
# for the full configuration with documentation.
#
# Re-exported here so all config is accessible from one import:
#   from config.settings import REMOTE_INFERENCE_ENABLED, REMOTE_INFERENCE_URL
# ---------------------------------------------------------------------------
from generation.inference_config import (  # noqa: E402
    REMOTE_INFERENCE_ENABLED,
    REMOTE_INFERENCE_URL,
    REMOTE_TIMEOUT,
    REMOTE_STREAMING,
    REMOTE_MAX_TOKENS,
    REMOTE_TEMPERATURE,
    RESPONSE_CACHE_ENABLED,
)

