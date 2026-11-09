# 🤖 Phase 4: Generative AI & LLM Engineering

> **Duration**: 6–8 weeks · **Difficulty**: 🟠–🔴 Advanced to Expert · **Prerequisites**: Phase 1 (Foundations), Phase 2 (ML Fundamentals), & Phase 3 (Deep Learning)  
> **⭐ Core Milestone**: This is the practical core of the 2026 AI Engineer role, focusing on LLM orchestration, semantic retrieval, autonomous agentic systems, and preference alignment.

---

## 📓 Notebook Curriculum

| # | Notebook | Domain | Key Concepts | Status |
|:-:|----------|--------|--------------|:------:|
| **01** | [`01-prompt-engineering.ipynb`](01-prompt-engineering.ipynb) | Prompting & Logic | In-Context Learning, Zero/Few-Shot, Chain-of-Thought (CoT), Step-Back, ReAct, Pydantic structured output validation, injection defense | ✅ Complete |
| **02** | [`02-local-llms-inference.ipynb`](02-local-llms-inference.ipynb) | Local Inference | Open-weights LLMs, KV caching mechanics, sampling (Temperature, Top-k, Top-p, Min-p), PTQ quantization (GGUF, AWQ, NF4), streaming | ✅ Complete |
| **03** | [`03-embeddings-vector-dbs.ipynb`](03-embeddings-vector-dbs.ipynb) | Vector Search | Dense embeddings (`sentence-transformers`), metric spaces, FAISS index architectures (`IndexFlatIP`, `IndexIVFFlat`, `IndexHNSWFlat`), recall vs latency | ✅ Complete |
| **04** | [`04-rag-fundamentals.ipynb`](04-rag-fundamentals.ipynb) | RAG Foundations | Document loaders, chunking strategies (recursive, semantic), vector store retrieval, context injection, citation attribution, RAGAS metrics | ✅ Complete |
| **05** | [`05-advanced-rag-patterns.ipynb`](05-advanced-rag-patterns.ipynb) | Advanced RAG | Hybrid search (BM25 + Dense Reciprocal Rank Fusion), Cross-Encoder re-ranking, Semantic Routing, HyDE query expansion, GraphRAG triplets | ✅ Complete |
| **06** | [`06-ai-agents-tool-use.ipynb`](06-ai-agents-tool-use.ipynb) | Autonomous Agents | Pure Python ReAct execution loop, JSON Schema tool calling, custom sandboxes (Math/Code/Search), error reflection, stateful memory buffer | ✅ Complete |
| **07** | [`07-multi-agent-systems.ipynb`](07-multi-agent-systems.ipynb) | Multi-Agent Systems | Stateful directed graphs (`LangGraph`), supervisor orchestration, role-based teams, adversarial debate & consensus, human-in-the-loop breakpoints | ✅ Complete |
| **08** | [`08-fine-tuning-lora.ipynb`](08-fine-tuning-lora.ipynb) | Fine-Tuning | Parameter-Efficient Fine-Tuning (PEFT), LoRA rank decomposition ($W_0 + \frac{\alpha}{r} B A$), QLoRA NF4, HuggingFace `SFTTrainer`, weight merging | ✅ Complete |
| **09** | [`09-dpo-alignment.ipynb`](09-dpo-alignment.ipynb) | Alignment | RLHF vs Direct Preference Optimization (DPO), Bradley-Terry preference model, pairwise datasets, implicit reward margin, HuggingFace `DPOTrainer` | ✅ Complete |

---

## 🗺️ Architectural Workflow

```
                                  [User Natural Language Query]
                                                |
                                                v
               +-----------------------------------------------------------------+
               |              Prompt Engineering & Query Routing                 |
               |       - Query Reformulation / HyDE (Notebook 01 & 05)           |
               |       - Intent Classification & Schema Binding                  |
               +-----------------------------------------------------------------+
                                       /                     \
                                      /                       \
                                     v                         v
        +-----------------------------------------+   +---------------------------------+
        |           Semantic Retrieval            |   |      Autonomous Tool Engine     |
        | - Dense Embeddings + BM25 Hybrid (03,05)|   | - ReAct Execution Loop (06)     |
        | - Cross-Encoder Re-Ranking (05)         |   | - Multi-Agent Teams (07)        |
        | - Context Injection & Attribution (04)  |   | - Dynamic Sandboxes & APIs      |
        +-----------------------------------------+   +---------------------------------+
                                      \                       /
                                       \                     /
                                        v                   v
               +-----------------------------------------------------------------+
               |                 Local Generative LLM Backbone                   |
               |       - High-Throughput Inference with KV Caching (02)          |
               |       - Domain-Adapted with LoRA / QLoRA (08)                   |
               |       - Safety & Preference Aligned with DPO (09)               |
               +-----------------------------------------------------------------+
                                                |
                                                v
                                [Grounded, Structured Response]
```

---

## 🛠️ Packages Used

```python
torch >= 2.0.0
transformers >= 4.40.0
datasets >= 2.18.0
peft >= 0.10.0
trl >= 0.8.0
sentence-transformers >= 2.6.0
faiss-cpu >= 1.8.0
pydantic >= 2.6.0
langchain-core >= 0.1.30
langgraph >= 0.0.30
```

---

**[← Notebooks Index](../README.md)** · **[← Phase 3: Deep Learning](../03-deep-learning/)** · **[Next: Phase 5: MLOps & LLMOps →](../05-mlops-llmops/)**
