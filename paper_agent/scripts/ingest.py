"""
CLI Script — Ingest a PDF into the FAISS index from the command line.

Usage:
    python scripts/ingest.py path/to/paper.pdf
    python scripts/ingest.py path/to/paper.pdf --strip-refs
"""

import sys
import argparse
import logging
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import ensure_directories
from ingestion.pipeline import ingest_pdf
from embeddings.embedder import get_embedding_model
from vectorstore.faiss_store import (
    create_index,
    save_index,
    load_index,
    add_documents,
    index_exists,
    get_index_stats,
)


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into the Paper Tutor index")
    parser.add_argument("pdf_path", help="Path to the PDF file to ingest")
    parser.add_argument(
        "--strip-refs", action="store_true",
        help="Remove the References section before indexing",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ensure_directories()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    # Step 1: Ingest
    print(f"\n📄 Ingesting: {pdf_path.name}")
    result = ingest_pdf(pdf_path, strip_references=args.strip_refs)

    chunks = result["chunks"]
    meta = result["metadata"]
    stats = result["stats"]

    print(f"   Title:  {meta['title']}")
    print(f"   Pages:  {meta['pages']}")
    print(f"   Chunks: {stats['total_chunks']}")
    print(f"   Avg:    ~{stats.get('avg_size_tokens', '?')} tokens/chunk")

    # Step 2: Embed and index
    print(f"\n🔢 Embedding {len(chunks)} chunks...")
    embedding_model = get_embedding_model()

    if index_exists():
        print("   Loading existing index...")
        index = load_index(embedding_model=embedding_model)
        add_documents(index, chunks)
    else:
        print("   Creating new index...")
        index = create_index(chunks, embedding_model=embedding_model)

    # Step 3: Save
    save_index(index)
    idx_stats = get_index_stats(index)

    print(f"\n✅ Done! Index: {idx_stats['total_vectors']} total vectors")
    print(f"   Run the UI:  streamlit run ui/app.py\n")


if __name__ == "__main__":
    main()
