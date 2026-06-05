# Generation package — LLM client, prompts, and chain orchestration
#
# Public API:
#   PaperTutorChain(index) — main RAG chain
#   get_llm() — cached Ollama LLM client
#   get_prompt(mode) — prompt template selector
#   check_ollama_connection() — health check
#
from generation.chain import PaperTutorChain, ChainResponse
from generation.llm_client import get_llm, check_ollama_connection
from generation.prompts import get_prompt

__all__ = [
    "PaperTutorChain",
    "ChainResponse",
    "get_llm",
    "check_ollama_connection",
    "get_prompt",
]
