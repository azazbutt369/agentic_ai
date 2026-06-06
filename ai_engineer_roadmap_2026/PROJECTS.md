<div align="center">

# 💻 Capstone Projects

**7 production-ready projects that teach real AI engineering skills and look great on your resume.**

*Every project uses 100% free, open-source tools. No API keys required.*

</div>

---

## Project Overview

| # | Project | Difficulty | Time | Key Skills | Roadmap Phase |
|:-:|---------|:----------:|:----:|------------|:-------------:|
| 1 | [Semantic Search Engine](#1-semantic-search-engine) | ⭐⭐ | ~8h | Embeddings, Vector DB, FastAPI | Phase 4 |
| 2 | [RAG Chatbot](#2-rag-chatbot) | ⭐⭐⭐ | ~16h | RAG, LangChain, Ollama, Streaming | Phase 4 |
| 3 | [AI Agent Framework](#3-ai-agent-framework) | ⭐⭐⭐⭐ | ~24h | Tool Use, ReAct, Memory, Multi-Agent | Phase 4 |
| 4 | [LLM Fine-Tuning Pipeline](#4-llm-fine-tuning-pipeline) | ⭐⭐⭐ | ~16h | LoRA, QLoRA, PEFT, HuggingFace TRL | Phase 4 |
| 5 | [Model Evaluation Suite](#5-model-evaluation-suite) | ⭐⭐⭐ | ~12h | Benchmarking, Metrics, Dashboard | Phase 4–5 |
| 6 | [Multimodal Content Analyzer](#6-multimodal-content-analyzer) | ⭐⭐⭐⭐ | ~20h | Vision-Language, Image Understanding | Phase 6 |
| 7 | [Production LLM API](#7-production-llm-api) | ⭐⭐⭐⭐⭐ | ~32h | vLLM, Docker, Auth, Monitoring | Phase 5 |

### Recommended Project Path

```
Beginner          Intermediate              Advanced                Expert
   │                   │                       │                      │
   └──▶ Project 1 ──▶ Project 2 ──▶ Project 4 ──▶ Project 3 ──▶ Project 7
                       │                       │
                       └──▶ Project 5 ──────▶ Project 6
```

---

## 1. Semantic Search Engine

> **Build a search engine that understands meaning, not just keywords.**

### 🎯 What You'll Build

A FastAPI-powered search engine that converts documents into vector embeddings and performs semantic similarity search. Users can search a knowledge base using natural language queries and get contextually relevant results — even when no keywords match.

### 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| Embeddings | `sentence-transformers` (BAAI/bge-small-en-v1.5) |
| Vector Database | Chroma |
| API Framework | FastAPI |
| Frontend | HTML + HTMX (minimal, functional UI) |
| Data | Wikipedia articles (HuggingFace `wikipedia` dataset) |

### 📐 Architecture

```
┌──────────┐     ┌──────────────┐     ┌──────────┐     ┌──────────┐
│  User    │────▶│   FastAPI    │────▶│ Embedding│────▶│  Chroma  │
│  Query   │     │   Server     │     │  Model   │     │  Vector  │
└──────────┘     └──────────────┘     └──────────┘     │    DB    │
                        │                               └──────────┘
                        │                                     │
                        │◀────────── Top-K Results ──────────┘
                        │
                 ┌──────────────┐
                 │   Response   │
                 │  (Ranked)    │
                 └──────────────┘
```

### 📦 Project Structure

```
projects/01-semantic-search-engine/
├── README.md               # Setup, usage, architecture
├── requirements.txt         # sentence-transformers, chromadb, fastapi, uvicorn
├── architecture.md          # Detailed design document
├── src/
│   ├── __init__.py
│   ├── main.py             # FastAPI application
│   ├── embeddings.py       # Embedding model wrapper
│   ├── vectorstore.py      # Chroma DB operations
│   ├── indexer.py          # Document indexing pipeline
│   └── config.py           # Configuration management
├── tests/
│   ├── test_embeddings.py
│   ├── test_vectorstore.py
│   └── test_api.py
├── templates/
│   └── search.html         # Simple search UI
└── assets/
    └── architecture.png
```

### 🎓 What You'll Learn
- How text embeddings capture semantic meaning
- Vector database indexing and similarity search
- Building production REST APIs with FastAPI
- Batch processing and indexing pipelines

### 💼 Portfolio Value
> *"Built a semantic search engine with sub-100ms query latency over 100K+ documents using vector embeddings and FastAPI."* — This directly maps to how modern search at companies like Notion, Slack, and Confluence works.

---

## 2. RAG Chatbot

> **Build a chatbot that answers questions from your own documents — no cloud APIs required.**

### 🎯 What You'll Build

A Retrieval-Augmented Generation chatbot that ingests PDFs, Markdown files, or web pages, chunks them intelligently, stores them in a vector database, and uses a local LLM (via Ollama) to generate grounded, cited answers with streaming responses.

### 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| LLM | Ollama (Llama 3 / Mistral) |
| Embeddings | `sentence-transformers` (BAAI/bge) |
| Vector Database | Chroma |
| Framework | LangChain |
| UI | Gradio |
| Document Parsing | `unstructured`, `pypdf` |

### 📐 Architecture

```
┌────────┐    ┌───────────┐    ┌──────────┐    ┌───────────┐
│  Docs  │───▶│  Chunker  │───▶│ Embedder │───▶│  Chroma   │
│ (PDF,  │    │ (Recursive│    │ (BGE)    │    │ (Persist) │
│  MD)   │    │  Splitter)│    └──────────┘    └───────────┘
└────────┘    └───────────┘                         │
                                                    │
┌────────┐    ┌───────────┐    ┌──────────┐         │
│  User  │───▶│  Retriever│───▶│  Ollama  │◀────────┘
│ Query  │    │ + Reranker│    │   LLM    │    Retrieved
└────────┘    └───────────┘    └──────────┘    Context
                                    │
                            ┌───────────────┐
                            │   Streaming   │
                            │   Response    │
                            │  (+ Sources)  │
                            └───────────────┘
```

### 📦 Project Structure

```
projects/02-rag-chatbot/
├── README.md
├── requirements.txt
├── architecture.md
├── src/
│   ├── __init__.py
│   ├── app.py              # Gradio chat interface
│   ├── ingestion.py        # Document loading & chunking
│   ├── retriever.py        # Vector search + reranking
│   ├── chain.py            # LangChain RAG chain
│   ├── prompts.py          # System & user prompt templates
│   └── config.py
├── tests/
│   ├── test_ingestion.py
│   ├── test_retriever.py
│   └── test_chain.py
├── sample_docs/            # Example documents for testing
│   ├── sample_paper.pdf
│   └── sample_notes.md
└── assets/
    └── demo.gif
```

### 🎓 What You'll Learn
- End-to-end RAG pipeline architecture
- Document chunking strategies and their tradeoffs
- Retrieval with reranking for improved relevance
- Streaming LLM responses for real-time UX
- Prompt engineering for grounded, cited answers

### 💼 Portfolio Value
> *"Built a document Q&A chatbot using RAG with local LLMs, achieving cited responses from private knowledge bases with streaming output."* — RAG is the #1 most in-demand skill for AI Engineers in 2026.

---

## 3. AI Agent Framework

> **Build autonomous AI agents that reason, plan, use tools, and collaborate.**

### 🎯 What You'll Build

A modular agent framework that implements the ReAct pattern (Reasoning + Acting). Agents can use custom tools (web search, code execution, file operations), maintain memory across conversations, and collaborate in multi-agent teams to solve complex tasks.

### 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| LLM | Ollama (Llama 3 / Mistral) |
| Agent Framework | LangGraph + Custom |
| Tools | DuckDuckGo Search, Python REPL, File I/O |
| Memory | SQLite (persistent), in-memory (session) |
| UI | Gradio |

### 📐 Architecture

```
┌─────────────────────────────────────────────────┐
│                AGENT FRAMEWORK                   │
│                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Agent 1  │◀──▶│ Orchestr │◀──▶│  Agent 2  │  │
│  │(Researcher│    │  -ator   │    │(Writer)   │  │
│  └──────────┘    └──────────┘    └──────────┘  │
│       │                                  │       │
│  ┌──────────┐                      ┌──────────┐│
│  │  Tools   │                      │  Memory  ││
│  │• Search  │                      │• Short   ││
│  │• Code    │                      │• Long    ││
│  │• Files   │                      │  Term    ││
│  └──────────┘                      └──────────┘│
└─────────────────────────────────────────────────┘
```

### 📦 Project Structure

```
projects/03-ai-agent-framework/
├── README.md
├── requirements.txt
├── architecture.md
├── src/
│   ├── __init__.py
│   ├── agent.py            # Base agent class (ReAct pattern)
│   ├── orchestrator.py     # Multi-agent coordinator
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py         # Tool base class
│   │   ├── search.py       # DuckDuckGo search tool
│   │   ├── code_exec.py    # Python code execution (sandboxed)
│   │   └── file_ops.py     # File read/write operations
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── short_term.py   # Conversation buffer
│   │   └── long_term.py    # SQLite-backed persistent memory
│   ├── prompts.py          # Agent system prompts
│   ├── app.py              # Gradio demo
│   └── config.py
├── tests/
│   ├── test_agent.py
│   ├── test_tools.py
│   ├── test_memory.py
│   └── test_orchestrator.py
└── assets/
    └── agent_flow.png
```

### 🎓 What You'll Learn
- ReAct pattern for reasoning and acting
- Tool/function calling with structured outputs
- Memory architectures (buffer, summary, vector-backed)
- Multi-agent orchestration patterns
- Sandboxed code execution for safety

### 💼 Portfolio Value
> *"Designed and built a multi-agent AI framework with tool use, persistent memory, and inter-agent collaboration using LangGraph."* — Agent engineering is the fastest-growing specialization in AI.

---

## 4. LLM Fine-Tuning Pipeline

> **Fine-tune open-weight LLMs on your own data using consumer hardware.**

### 🎯 What You'll Build

An end-to-end pipeline for fine-tuning open-weight models (Llama, Mistral, Phi) on custom instruction datasets. Uses LoRA/QLoRA for parameter-efficient training on consumer GPUs (8GB+ VRAM). Includes dataset preparation, training, evaluation, and model export.

### 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| Base Models | HuggingFace Hub (Llama, Mistral, Phi) |
| Fine-Tuning | HuggingFace TRL, PEFT |
| Quantization | bitsandbytes (QLoRA) |
| Tracking | Weights & Biases (free tier) or TensorBoard |
| Dataset | HuggingFace Datasets |
| Export | GGUF conversion for Ollama |

### 📦 Project Structure

```
projects/04-llm-fine-tuning-pipeline/
├── README.md
├── requirements.txt
├── architecture.md
├── src/
│   ├── __init__.py
│   ├── data_prep.py        # Dataset loading & formatting
│   ├── train.py            # Training script (LoRA/QLoRA)
│   ├── evaluate.py         # Model evaluation & comparison
│   ├── export.py           # GGUF conversion for local inference
│   ├── config.py           # Training hyperparameters
│   └── utils.py            # Helpers (logging, metrics)
├── configs/
│   ├── qlora_llama.yaml    # QLoRA config for Llama
│   ├── lora_mistral.yaml   # LoRA config for Mistral
│   └── lora_phi.yaml       # LoRA config for Phi
├── tests/
│   ├── test_data_prep.py
│   ├── test_train.py
│   └── test_export.py
└── assets/
    ├── training_curves.png
    └── evaluation_results.png
```

### 🎓 What You'll Learn
- Parameter-Efficient Fine-Tuning (PEFT) theory and practice
- LoRA/QLoRA: training large models on consumer GPUs
- Instruction dataset formatting (Alpaca, ShareGPT, ChatML)
- Evaluation strategies for fine-tuned models
- Model export and deployment (GGUF for Ollama)

### 💼 Portfolio Value
> *"Built an end-to-end LLM fine-tuning pipeline using QLoRA on consumer GPUs, with automated evaluation and GGUF export for local deployment."* — Fine-tuning is how companies customize foundation models for their specific use cases.

---

## 5. Model Evaluation Suite

> **Build an automated benchmarking system for comparing LLM performance.**

### 🎯 What You'll Build

A comprehensive evaluation framework that benchmarks LLMs across multiple dimensions: accuracy, latency, cost (token usage), safety, and task-specific performance. Includes a dashboard for visualizing results and comparing models side-by-side.

### 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| LLMs | Ollama (multiple models) |
| Evaluation | Custom metrics + `lm-evaluation-harness` |
| Dashboard | Streamlit |
| Data | HuggingFace Datasets (MMLU, TruthfulQA subsets) |
| Visualization | Plotly, Matplotlib |

### 📦 Project Structure

```
projects/05-model-evaluation-suite/
├── README.md
├── requirements.txt
├── architecture.md
├── src/
│   ├── __init__.py
│   ├── runner.py           # Benchmark runner
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── accuracy.py     # Task accuracy metrics
│   │   ├── latency.py      # Response time metrics
│   │   ├── safety.py       # Safety/toxicity scoring
│   │   └── cost.py         # Token usage tracking
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── qa.py           # Question-answering benchmarks
│   │   ├── reasoning.py    # Reasoning benchmarks
│   │   └── coding.py       # Code generation benchmarks
│   ├── dashboard.py        # Streamlit leaderboard
│   └── config.py
├── tests/
│   ├── test_runner.py
│   └── test_metrics.py
└── assets/
    └── leaderboard_demo.png
```

### 🎓 What You'll Learn
- LLM evaluation methodology and common benchmarks
- Designing fair, reproducible model comparisons
- Building interactive dashboards with Streamlit
- Statistical significance in model comparisons

### 💼 Portfolio Value
> *"Built an automated LLM evaluation suite that benchmarks models across accuracy, latency, safety, and cost, with an interactive leaderboard dashboard."*

---

## 6. Multimodal Content Analyzer

> **Build an AI system that understands both text and images.**

### 🎯 What You'll Build

A pipeline that processes documents containing both text and images, extracts information from each modality, and provides unified analysis. Can describe images, extract text from screenshots, answer questions about visual content, and generate structured summaries.

### 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| Vision-Language Model | LLaVA / Ollama multimodal |
| Image Processing | Pillow, OpenCV |
| OCR (fallback) | Tesseract |
| API | FastAPI |
| Frontend | Gradio |

### 📦 Project Structure

```
projects/06-multimodal-content-analyzer/
├── README.md
├── requirements.txt
├── architecture.md
├── src/
│   ├── __init__.py
│   ├── app.py              # Gradio interface
│   ├── analyzer.py         # Core multimodal analysis pipeline
│   ├── image_processor.py  # Image preprocessing & feature extraction
│   ├── text_processor.py   # Text extraction & NLP
│   ├── models.py           # Model loading & inference
│   └── config.py
├── tests/
│   ├── test_analyzer.py
│   ├── test_image.py
│   └── test_text.py
├── sample_data/
│   ├── sample_document.pdf
│   └── sample_images/
└── assets/
    └── demo.gif
```

### 🎓 What You'll Learn
- Vision-Language model architectures
- Processing interleaved multimodal inputs
- Image preprocessing pipelines
- Building unified text + image analysis workflows

### 💼 Portfolio Value
> *"Built a multimodal content analyzer combining vision-language models for unified text and image understanding."* — Multimodal AI is the fastest-growing frontier in 2026.

---

## 7. Production LLM API

> **Build a production-grade LLM serving API with authentication, rate limiting, monitoring, and Docker deployment.**

### 🎯 What You'll Build

The ultimate capstone: a production-ready API that serves LLM inference at scale. Features include token-based authentication, per-user rate limiting, request queuing, streaming responses, health checks, Prometheus metrics, and full Docker deployment.

### 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| Model Serving | vLLM |
| API Framework | FastAPI |
| Authentication | JWT (PyJWT) |
| Rate Limiting | slowapi |
| Monitoring | Prometheus + Grafana |
| Containerization | Docker + Docker Compose |
| Load Testing | Locust |

### 📐 Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              Docker Compose                  │
                    │                                              │
┌──────────┐       │  ┌──────────┐    ┌──────────┐              │
│  Client  │──────▶│  │  Nginx   │───▶│  FastAPI  │              │
│          │       │  │  Proxy   │    │  Gateway  │              │
└──────────┘       │  └──────────┘    └──────────┘              │
                    │                       │                     │
                    │        ┌──────────────┼──────────────┐     │
                    │        │              │              │     │
                    │   ┌─────────┐  ┌──────────┐  ┌──────────┐│
                    │   │  Auth   │  │   vLLM   │  │Prometheus││
                    │   │  (JWT)  │  │  Server  │  │ + Grafana││
                    │   └─────────┘  └──────────┘  └──────────┘│
                    └─────────────────────────────────────────────┘
```

### 📦 Project Structure

```
projects/07-production-llm-api/
├── README.md
├── requirements.txt
├── architecture.md
├── Dockerfile                # Multi-stage production build
├── docker-compose.yml        # Full stack deployment
├── nginx.conf                # Reverse proxy config
├── prometheus.yml            # Monitoring config
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── auth.py              # JWT authentication
│   ├── rate_limiter.py      # Per-user rate limiting
│   ├── inference.py         # vLLM inference client
│   ├── streaming.py         # SSE streaming responses
│   ├── health.py            # Health check endpoints
│   ├── metrics.py           # Prometheus metrics
│   ├── models.py            # Pydantic request/response models
│   └── config.py
├── tests/
│   ├── test_auth.py
│   ├── test_rate_limiter.py
│   ├── test_inference.py
│   └── test_api.py
├── loadtest/
│   └── locustfile.py        # Load testing scenarios
├── grafana/
│   └── dashboard.json       # Pre-built monitoring dashboard
└── assets/
    ├── grafana_dashboard.png
    └── load_test_results.png
```

### 🎓 What You'll Learn
- Production LLM serving with vLLM (PagedAttention, continuous batching)
- API security (JWT auth, rate limiting, input validation)
- Infrastructure as code (Docker Compose, Nginx)
- Observability (Prometheus metrics, Grafana dashboards)
- Load testing and performance optimization
- Production deployment best practices

### 💼 Portfolio Value
> *"Designed and deployed a production LLM API serving 100+ concurrent users with JWT auth, rate limiting, Prometheus monitoring, and Grafana dashboards — fully containerized with Docker Compose."* — This is exactly what companies need from a Senior AI Engineer.

---

## How Projects Are Evaluated

Every project in this repository includes an evaluation rubric:

| Criterion | Weight | What We Look For |
|-----------|:------:|------------------|
| **Functionality** | 30% | Does it work? Does it handle edge cases? |
| **Code Quality** | 25% | Clean code, type hints, error handling, documentation |
| **Architecture** | 20% | Modular design, separation of concerns, extensibility |
| **Testing** | 15% | Unit tests, integration tests, edge case coverage |
| **Documentation** | 10% | README, docstrings, architecture diagrams |

---

<div align="center">

**[← Back to README](README.md)** · **[Roadmap →](ROADMAP.md)** · **[Resources →](RESOURCES.md)**

</div>
