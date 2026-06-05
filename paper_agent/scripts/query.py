"""
CLI Script — Query the Paper Tutor from the command line.

Usage:
    python scripts/query.py "How does the Transformer work?"
    python scripts/query.py "Summarize the paper" --mode summarize
    python scripts/query.py "Compare methods" --mode compare
"""

import sys
import argparse
import logging
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from embeddings.embedder import get_embedding_model
from vectorstore.faiss_store import load_index, index_exists
from generation.chain import PaperTutorChain
from memory.chat_memory import ChatMemory


def main():
    parser = argparse.ArgumentParser(description="Query the Paper Tutor Agent")
    parser.add_argument("question", help="Your question about the papers")
    parser.add_argument(
        "--mode", choices=["qa", "summarize", "compare"],
        default="qa", help="Query mode (default: qa)",
    )
    parser.add_argument(
        "--source", default=None,
        help="Restrict to a specific source file",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    if not index_exists():
        print("❌ No papers indexed yet. Run: python scripts/ingest.py <pdf>")
        sys.exit(1)

    # Load index
    print("Loading index...")
    embedding_model = get_embedding_model()
    index = load_index(embedding_model=embedding_model)

    # Create chain
    chain = PaperTutorChain(
        index=index,
        memory=ChatMemory(),
        mode=args.mode,
    )

    # Query
    print(f"\n❓ {args.question}\n")
    response = chain.ask(
        question=args.question,
        mode=args.mode,
        source_filter=args.source,
    )

    if response.error:
        print(f"❌ Error: {response.error}")
        sys.exit(1)

    # Display answer
    print(f"🤖 {response.answer}\n")

    # Display sources
    if response.sources:
        print("📎 Sources:")
        for src in response.sources:
            score_pct = int(src["score"] * 100)
            print(f"   📄 {src['source']} · Page {src['page']} · {score_pct}% match")

    print()


if __name__ == "__main__":
    main()
