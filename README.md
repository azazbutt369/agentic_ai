<div align="center">

# 🤖 Agentic AI: Production Agents & AI Engineering 2026

**A comprehensive, free, production-grade ecosystem for Agentic Systems, Local RAG, and AI Engineering.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C.svg?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_Inference-000000.svg?logo=ollama&logoColor=white)](https://ollama.ai/)
[![FAISS](https://img.shields.io/badge/FAISS-VectorStore-0080FF.svg?logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![100% Free](https://img.shields.io/badge/Cost-100%25_Free_&_Local-00C853.svg)](#-core-principles)

<p align="center">
  <a href="#-repository-modules">Modules</a> •
  <a href="#-flagship-project-ml-paper-tutor-agent">ML Paper Tutor Agent</a> •
  <a href="#-ai-engineer-roadmap-2026">2026 Curriculum</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-contributing">Contributing</a>
</p>

</div>

---

## 📌 Repository Modules

This repository is split into two complementary production systems:

| Module | Purpose | Key Technologies | Link |
|---|---|---|:---:|
| 📄🤖 **`paper_agent/`** | **ML Paper Tutor Agent**: Production-style, 100% local RAG agent that ingests research papers (PDFs), chunks text, generates embeddings, retrieves context via FAISS, and answers questions with source citations. | `LangChain`, `Qwen2.5-1.5B`, `Ollama`, `FAISS`, `bge-small`, `Streamlit` | [**Explore Agent →**](paper_agent/) |
| 🗺️📓 **`ai_engineer_roadmap_2026/`** | **AI Engineer Roadmap 2026**: A 35-notebook hands-on curriculum spanning 6 rigorous phases (Foundations, ML Fundamentals, Deep Learning, GenAI, MLOps, Advanced Paradigms). | `PyTorch`, `scikit-learn`, `Optuna`, `XGBoost`, `LightGBM` | [**Explore Curriculum →**](ai_engineer_roadmap_2026/) |

---

## 📄🤖 Flagship Project: ML Paper Tutor Agent

An autonomous research paper assistant designed to run **entirely on consumer hardware without cloud dependencies, API keys, or costs**.

```
User Query / PDF Upload
         │
         ▼
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│ Streamlit Web UI │ ────▶ │ Document Chunker │ ────▶ │ bge-small-en-v1  │
│  (Modern Dark)   │       │ (Recursive/Char) │       │   (384-dim)      │
└────────┬─────────┘       └──────────────────┘       └────────┬─────────┘
         │                                                     │
         ▼                                                     ▼
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│ Conversational   │ ◀──── │  Context Merger  │ ◀──── │   FAISS Index    │
│ Memory + Prompt  │       │  & Citations     │       │   (Fast Search)  │
└────────┬─────────┘       └──────────────────┘       └──────────────────┘
         │
         ▼
┌──────────────────┐
│  Ollama / Qwen   │ ────▶ Streamed Response with [File.pdf, Page X] Citations
│  2.5-1.5B Local  │
└──────────────────┘
```

### ✨ Key Features
- ⚡ **100% Local Inference**: Runs on Qwen2.5-1.5B-Instruct via Ollama (requires < 4GB RAM).
- 🔍 **Dense Semantic Retrieval**: Employs `bge-small-en-v1.5` embeddings with FAISS vector similarity search.
- 📎 **Citation-Aware QA**: Every claim is cited directly to source PDF page numbers.
- 💬 **Conversational Memory**: Maintains multi-turn context with chat history sliding windows.
- 🎨 **Streamlit Frontend**: Clean dark-mode UI with document upload, status progress, and interactive chat.
- ☁️ **Google Colab Server Mode**: Includes tunnel server configuration for running GPU inference on free Colab instances.

---

## 🗺️📓 AI Engineer Roadmap 2026

A 35-notebook executable curriculum adhering to the **10-Section Production Standard**:
1. Overview • 2. Objectives • 3. Versioned Imports • 4. Config & Seeds • 5. LaTeX Theory • 6. Implementation • 7. Visual Evaluation • 8. Guided Exercises • 9. Portfolio Challenges • 10. Further Reading.

```
Phase 1: Foundations (7 Notebooks) ──▶ Math, Python Mastery, NumPy, Pandas/Polars, Git
               │
               ▼
Phase 2: ML Fundamentals (6 Notebooks) ──▶ Supervised, Unsupervised, Metrics, Pipelines, Feature Engineering, Tuning
               │
               ▼
Phase 3: Deep Learning (6 Notebooks) ──▶ PyTorch Internals, CNNs, Transformers from scratch
               │
               ▼
Phase 4: Generative AI (9 Notebooks) ──▶ LLMs, RAG, Agentic Systems, Fine-tuning, DPO
               │
               ▼
Phase 5: MLOps & LLMOps (4 Notebooks) ──▶ Docker, vLLM, Observability, CI/CD
               │
               ▼
Phase 6: Advanced Paradigms (3 Notebooks) ──▶ Multimodal, Edge AI, Safety & Alignment
```

### 📊 Phase 2 Notebooks Index (Complete & Executed)
- [`01-supervised-learning.ipynb`](ai_engineer_roadmap_2026/notebooks/02-ml-fundamentals/01-supervised-learning.ipynb) — Classification & Regression (Trees, Ensembles, XGBoost, LightGBM)
- [`02-unsupervised-learning.ipynb`](ai_engineer_roadmap_2026/notebooks/02-ml-fundamentals/02-unsupervised-learning.ipynb) — Clustering & Dimensionality Reduction (K-Means, PCA, t-SNE)
- [`03-evaluation-metrics.ipynb`](ai_engineer_roadmap_2026/notebooks/02-ml-fundamentals/03-evaluation-metrics.ipynb) — Accuracy Paradox, ROC vs. PR Curves, MCC, Regression Metrics
- [`04-scikit-learn-workflow.ipynb`](ai_engineer_roadmap_2026/notebooks/02-ml-fundamentals/04-scikit-learn-workflow.ipynb) — Pipelines, ColumnTransformer, Preventing Data Leakage, Joblib
- [`05-feature-engineering.ipynb`](ai_engineer_roadmap_2026/notebooks/02-ml-fundamentals/05-feature-engineering.ipynb) — Imputation, Power Transforms, Cyclical Encodings, Boruta Selection
- [`06-hyperparameter-tuning.ipynb`](ai_engineer_roadmap_2026/notebooks/02-ml-fundamentals/06-hyperparameter-tuning.ipynb) — Grid vs. Random, Optuna TPE, Pruning, Nested Cross-Validation

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/azazbutt369/agentic_ai.git
cd agentic_ai
```

### 2. Launch ML Paper Tutor Agent
```bash
cd paper_agent

# Install dependencies
pip install -r requirements.txt

# Start Ollama with Qwen2.5
ollama pull qwen2.5:1.5b
ollama serve &

# Launch Streamlit Interface
streamlit run ui/app.py
```

### 3. Explore Roadmap Notebooks
```bash
cd ../ai_engineer_roadmap_2026
pip install -r requirements.txt
jupyter lab
```

---

## 🛡️ Core Principles

- ✅ **100% Free & Open-Source**: Zero paid API dependencies (OpenAI, Anthropic).
- ✅ **Reproducible**: Pinned seeds, verified package versions, top-to-bottom execution.
- ✅ **Clean Production Code**: Type annotations, docstrings, modular design.
- ✅ **Real-World Signals**: No toy datasets; uses realistic distributions with real noise and edge cases.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Please check [`CONTRIBUTING.md`](ai_engineer_roadmap_2026/CONTRIBUTING.md) and feel free to open a pull request.

---

<div align="center">
  <sub>Maintained with ❤️ by <a href="https://github.com/azazbutt369">Azaz Butt</a></sub>
</div>
