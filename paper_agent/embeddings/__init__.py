# Embeddings package — text-to-vector encoding
#
# Public API:
#   get_embedding_model() → HuggingFaceEmbeddings (cached singleton)
#   embed_texts(texts) → list[list[float]]
#   embed_query(query) → list[float]
#
from embeddings.embedder import get_embedding_model, embed_texts, embed_query

__all__ = ["get_embedding_model", "embed_texts", "embed_query"]
