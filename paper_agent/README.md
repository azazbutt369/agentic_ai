# ML Paper Tutor Agent 📄🤖

An AI-powered research paper assistant that ingests ML papers (PDFs), understands their content through RAG (Retrieval-Augmented Generation), and answers questions — all running **locally and for free**.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **PDF Ingestion** | Upload ML papers → automatic parsing, cleaning, and chunking |
| 🔍 **Semantic Search** | Find relevant passages using dense vector similarity (FAISS) |
| 💬 **Conversational QA** | Ask questions with follow-up support and conversation memory |
| 📝 **Paper Summarization** | Get structured summaries with key contributions and methodology |
| ⚖️ **Method Comparison** | Compare approaches across multiple papers |
| 📎 **Citation-Aware** | Every answer includes source citations `[Source: file.pdf, Page X]` |
| 🖥️ **100% Local** | Runs entirely on your machine — no cloud APIs, no costs |
| 🎨 **Premium UI** | Beautiful Streamlit interface with dark theme |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Streamlit   │────▶│  Retriever   │────▶│  FAISS Index  │
│  Frontend    │     │  (MMR/Sim)   │     │  (384-dim)   │
└──────┬───────┘     └──────┬───────┘     └──────────────┘
       │                    │
       ▼                    ▼
┌──────────────┐     ┌──────────────┐
│  RAG Chain   │◀────│  Context +   │
│  (Prompts +  │     │  Citations   │
│   Memory)    │     └──────────────┘
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Ollama    │
│  Qwen2.5    │
│   1.5B      │
└──────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|---|---|---|
| **LLM** | Qwen2.5-1.5B-Instruct (Ollama) | Best quality-to-size ratio, runs on 4GB RAM |
| **Embeddings** | bge-small-en-v1.5 | 384-dim, top MTEB scores for its size (~130MB) |
| **Vector DB** | FAISS (CPU) | Zero infrastructure, millisecond search |
| **Framework** | LangChain | Batteries-included RAG toolkit |
| **Frontend** | Streamlit | Rapid UI with chat widgets and file upload |
| **Backend** | Python 3.10+ | Simple, no server framework needed |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **4 GB RAM** minimum (8 GB recommended)
- **~4 GB disk space** (model + embeddings + index)

### 1. Install Dependencies

```bash
cd paper_agent
pip install -r requirements.txt
```

### 2. Install & Start Ollama

Download from [ollama.com/download](https://ollama.com/download), then:

```bash
# Start the Ollama server
ollama serve

# Pull the Qwen model (~1 GB download, one-time)
ollama pull qwen2.5:1.5b

# Verify it's available
ollama list
```

### 3. Set Up Environment

```bash
# Copy the environment template
cp .env.example .env

# Edit .env if you need to change defaults (usually not necessary)
```

### 4. Launch the UI

```bash
streamlit run ui/app.py
```

The app opens at `http://localhost:8501`. Upload a PDF and start asking questions!

### Alternative: CLI Usage

```bash
# Ingest a paper
python scripts/ingest.py path/to/paper.pdf

# Ask a question
python scripts/query.py "How does self-attention work?"

# Summarize
python scripts/query.py "Summarize this paper" --mode summarize

# Compare methods
python scripts/query.py "Compare the attention mechanisms" --mode compare
```

---

## 📁 Project Structure

```
paper_agent/
├── config/
│   └── settings.py          # Central configuration (all tunable parameters)
│
├── ingestion/
│   ├── pdf_loader.py         # PyMuPDF: PDF → page-level Documents
│   ├── text_cleaner.py       # Fix hyphens, unicode, headers, whitespace
│   ├── chunker.py            # RecursiveCharacterTextSplitter (512 tok / 64 overlap)
│   └── pipeline.py           # End-to-end: Load → Clean → Chunk
│
├── embeddings/
│   └── embedder.py           # bge-small-en-v1.5 wrapper (singleton, auto GPU/CPU)
│
├── vectorstore/
│   └── faiss_store.py        # FAISS CRUD: create, save, load, add, search, delete
│
├── retrieval/
│   └── retriever.py          # Semantic search + MMR + score threshold + citations
│
├── generation/
│   ├── llm_client.py         # Ollama ChatLLM wrapper (singleton, health check)
│   ├── prompts.py            # QA / Summarize / Compare prompt templates
│   └── chain.py              # PaperTutorChain: retrieve → prompt → generate
│
├── memory/
│   └── chat_memory.py        # Sliding window conversation memory (k=6)
│
├── ui/
│   └── app.py                # Streamlit frontend (premium dark theme)
│
├── scripts/
│   ├── ingest.py             # CLI: ingest a PDF
│   └── query.py              # CLI: ask a question
│
├── tests/
│   ├── test_ingestion.py     # 41 tests — PDF loading, cleaning, chunking
│   ├── test_embeddings.py    # 28 tests — embeddings, FAISS index
│   ├── test_retrieval.py     # 31 tests — search, MMR, filtering, citations
│   └── test_generation.py    # 39 tests — prompts, memory, chain (mocked LLM)
│
├── data/
│   ├── papers/               # Uploaded PDFs (gitignored)
│   └── faiss_index/          # Persisted FAISS index (gitignored)
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_ingestion.py -v    # Phase 2: Ingestion
python -m pytest tests/test_embeddings.py -v   # Phase 3: Embeddings + FAISS
python -m pytest tests/test_retrieval.py -v    # Phase 4: Retrieval
python -m pytest tests/test_generation.py -v   # Phase 5: Generation (no Ollama needed)
```

> **Note:** Embedding tests download the bge-small model (~130 MB) on first run.
> Generation tests use a mocked LLM — no Ollama server required.

---

## ⚙️ Configuration

All parameters are in `config/settings.py` and overridable via `.env`:

| Parameter | Default | Description |
|---|---|---|
| `LLM_MODEL` | `qwen2.5:1.5b` | Ollama model identifier |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature (lower = less hallucination) |
| `LLM_NUM_CTX` | `2048` | Context window size in tokens |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `CHUNK_SIZE` | `512` | Chunk size in tokens |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks in tokens |
| `TOP_K` | `4` | Number of chunks to retrieve per query |
| `SIMILARITY_THRESHOLD` | `0.35` | Minimum similarity score (0-1) |
| `USE_MMR` | `true` | Use MMR for diverse retrieval |
| `MEMORY_WINDOW_SIZE` | `6` | Number of recent exchanges to keep |

---

## 🧠 Key Concepts Explained

### What is RAG?
**Retrieval-Augmented Generation** combines search with LLM generation. Instead of asking the LLM to recall information from training data (which causes hallucination), we:
1. **Retrieve** relevant text chunks from our documents
2. **Augment** the LLM's prompt with these chunks as context
3. **Generate** an answer grounded in the retrieved evidence

### Why Chunking Matters
LLMs have limited context windows. We can't feed a 20-page paper in one go. Chunking breaks the paper into small, self-contained pieces (~512 tokens) that capture individual ideas. The quality of chunks directly determines retrieval quality.

### Why FAISS Over a Full Database
FAISS is a pure vector search library — no server, no config, no overhead. For our use case (< 100k vectors), it's faster than any database and persists to just two files on disk.

### Why Temperature 0.3
With a 1.5B model, higher temperatures (0.7+) cause significant hallucination. Lower temperatures (0.0) make responses robotic and repetitive. 0.3 is the sweet spot — mostly deterministic but natural.

---

## 🐛 Troubleshooting

### "Cannot connect to Ollama"
```bash
# Make sure Ollama is running
ollama serve

# Check it's responding
curl http://localhost:11434/api/tags
```

### "Model not found"
```bash
# Pull the model
ollama pull qwen2.5:1.5b

# Verify it's listed
ollama list
```

### "FAISS dimension mismatch"
This happens when switching embedding models. Delete the old index:
```bash
# Delete and re-ingest
rm -rf data/faiss_index/*
python scripts/ingest.py path/to/paper.pdf
```

### "Out of memory"
- Reduce `LLM_NUM_CTX` in `.env` (try `1024`)
- Use a smaller model: `ollama pull qwen2.5:0.5b`
- Close other applications to free RAM

### Slow first query
The embedding model (~130 MB) loads on first use. Subsequent queries are fast because the model is cached in memory.

---

## 🚀 Scaling Roadmap

| Enhancement | Complexity | Impact |
|---|---|---|
| Upgrade to Qwen2.5-3B | Low | Better answer quality |
| Add bge-reranker-v2-m3 | Medium | Much better retrieval precision |
| Hybrid search (BM25 + dense) | Medium | Better keyword matching |
| Export Q&A as Markdown notes | Low | Study aid feature |
| FastAPI backend (multi-user) | Medium | Production deployment |
| Fine-tune on ML paper Q&A | High | Domain-specific accuracy |
| Table/figure extraction | High | Richer paper understanding |

---

## 📝 License

MIT License — free for personal and commercial use.

---

## 🙏 Acknowledgments

- [Ollama](https://ollama.com/) — Local LLM inference
- [LangChain](https://langchain.com/) — RAG framework
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [Qwen2.5](https://huggingface.co/Qwen) — Open-source LLM
- [BGE](https://huggingface.co/BAAI/bge-small-en-v1.5) — Embedding model
