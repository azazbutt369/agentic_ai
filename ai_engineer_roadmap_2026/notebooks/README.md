<div align="center">

# 📓 Notebook Curriculum

**35 executable Jupyter notebooks covering the complete AI Engineer learning path.**

Every notebook runs top-to-bottom, uses real datasets, and teaches production-quality skills.

</div>

---

## Notebook Index

| Phase | Notebooks | Duration | Difficulty | Description |
|-------|:---------:|----------|:----------:|-------------|
| [01 — Foundations](01-foundations/) | 7 | 3–4 weeks | 🟢 | Math, Python, NumPy, Pandas, Git |
| [02 — ML Fundamentals](02-ml-fundamentals/) | 6 | 3–4 weeks | 🟡 | Supervised/Unsupervised ML, scikit-learn |
| [03 — Deep Learning](03-deep-learning/) | 6 | 4–5 weeks | 🟠 | PyTorch, CNNs, RNNs, Transformers |
| [04 — Generative AI](04-generative-ai/) | 9 | 6–8 weeks | 🟠–🔴 | LLMs, RAG, Agents, Fine-Tuning, DPO |
| [05 — MLOps & LLMOps](05-mlops-llmops/) | 4 | 3–4 weeks | 🟠 | Docker, vLLM, Monitoring, CI/CD |
| [06 — Advanced Paradigms](06-advanced-paradigms/) | 3 | 2–3 weeks | 🔴 | Multimodal, Edge AI, Safety |
| **Total** | **35** | **22–28 wks** | | |

---

## Notebook Standard

Every notebook follows a consistent 10-section structure:

```
📓 Every Notebook Contains:
├── 1. Overview              — What this notebook covers
├── 2. Learning Objectives   — What you'll be able to do after
├── 3. Imports               — All dependencies with version pinning
├── 4. Configuration         — Seeds, device setup, hyperparameters
├── 5. Theory                — Intuitive explanations with LaTeX equations
├── 6. Implementation        — Production-quality code with type hints
├── 7. Evaluation            — Metrics, plots, analysis
├── 8. Exercises             — Guided practice problems
├── 9. Challenge Problems    — Open-ended, portfolio-worthy challenges
└── 10. Further Reading      — Papers, docs, videos
```

### Quality Guarantees

| Guarantee | Description |
|-----------|-------------|
| ✅ **Runs top-to-bottom** | No manual intervention needed |
| ✅ **Reproducible** | Random seeds set, dependency versions pinned |
| ✅ **Real datasets** | HuggingFace, torchvision, UCI — no toy data |
| ✅ **No placeholders** | Every cell produces real output |
| ✅ **Explained** | Every code block preceded by a markdown explanation |
| ✅ **100% Free** | No paid API keys required |

---

## How to Run Notebooks

### Local (Recommended)

```bash
# From the repository root
jupyter lab notebooks/01-foundations/01-linear-algebra-for-ai.ipynb
```

### Docker

```bash
docker compose up jupyter
# Open http://localhost:8888
```

### Google Colab

Click the "Open in Colab" badge at the top of any notebook.

> ⚠️ Some Phase 4+ notebooks require Ollama, which cannot run in Colab.

---

## Prerequisites by Phase

| Phase | Python Packages | External Tools |
|-------|----------------|----------------|
| Phase 1 | numpy, pandas, polars, matplotlib | — |
| Phase 2 | scikit-learn, xgboost, lightgbm, optuna | — |
| Phase 3 | torch, torchvision, torchinfo | — |
| Phase 4 | langchain, chromadb, sentence-transformers, transformers, peft, trl | Ollama |
| Phase 5 | fastapi, uvicorn, prometheus-client | Docker |
| Phase 6 | pillow, opencv-python | Ollama |

All packages are listed in the root `requirements.txt`.

---

<div align="center">

**[← Back to README](../README.md)** · **[Roadmap →](../ROADMAP.md)** · **[Projects →](../PROJECTS.md)**

</div>
