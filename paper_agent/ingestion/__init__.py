# Ingestion package — PDF loading, cleaning, and chunking
#
# Public API:
#   ingest_pdf(file_path) → dict with chunks, metadata, stats
#   save_uploaded_pdf(name, bytes) → Path
#
from ingestion.pipeline import ingest_pdf, save_uploaded_pdf

__all__ = ["ingest_pdf", "save_uploaded_pdf"]
