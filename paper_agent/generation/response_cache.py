"""
==============================================================================
Response Cache — LRU cache for repeated inference queries
==============================================================================

WHY CACHING:
    LLM inference is the slowest step in the pipeline (~2-10s per query).
    If a user asks the same question twice (or rephrases identically),
    we can return the cached response INSTANTLY instead of re-running
    inference.

DESIGN:
    - Cache key = hash(prompt_text) — captures the full prompt including
      context and chat history.
    - LRU eviction with configurable max size (default 64 entries).
    - Thread-safe via collections.OrderedDict.
    - Cache is session-scoped (in-memory, not persisted to disk).
==============================================================================
"""

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from generation.inference_config import RESPONSE_CACHE_SIZE, RESPONSE_CACHE_ENABLED

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """A cached inference response."""
    answer: str
    prompt_hash: str
    hit_count: int = 0


class ResponseCache:
    """
    LRU (Least Recently Used) response cache for inference results.

    Usage:
        cache = ResponseCache(max_size=64)
        cache.put("prompt text", "response text")
        result = cache.get("prompt text")  # Returns "response text" or None
    """

    def __init__(self, max_size: int | None = None, enabled: bool | None = None):
        self.max_size = max_size or RESPONSE_CACHE_SIZE
        self.enabled = enabled if enabled is not None else RESPONSE_CACHE_ENABLED
        self._cache: OrderedDict[str, CachedResponse] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _hash_prompt(prompt: str) -> str:
        """Create a deterministic hash key from the prompt text."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    def get(self, prompt: str) -> Optional[str]:
        """
        Look up a cached response for the given prompt.

        Returns the cached answer string, or None if not found.
        """
        if not self.enabled:
            return None

        key = self._hash_prompt(prompt)

        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._cache[key].hit_count += 1
            self._hits += 1
            logger.debug(f"Cache HIT (key={key[:8]}...)")
            return self._cache[key].answer

        self._misses += 1
        return None

    def put(self, prompt: str, answer: str) -> None:
        """
        Store a response in the cache.

        If the cache is full, the least recently used entry is evicted.
        """
        if not self.enabled:
            return

        key = self._hash_prompt(prompt)

        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key].answer = answer
        else:
            if len(self._cache) >= self.max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug(f"Cache EVICT (key={evicted_key[:8]}...)")

            self._cache[key] = CachedResponse(
                answer=answer,
                prompt_hash=key,
            )

        logger.debug(f"Cache PUT (key={key[:8]}..., size={len(self._cache)})")

    def clear(self) -> None:
        """Clear all cached responses."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.debug("Cache cleared")

    @property
    def stats(self) -> dict:
        """Cache statistics for monitoring."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
            "enabled": self.enabled,
        }
