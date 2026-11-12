# ⚙️ Phase 5: MLOps & LLMOps

> **Duration**: 3–4 weeks · **Difficulty**: 🟠 Advanced · **Prerequisites**: Phase 1 (Foundations), Phase 2 (ML Fundamentals), Phase 3 (Deep Learning), & Phase 4 (Generative AI)  
> **⭐ Production Bridge**: Building a model is only half the job — this phase covers containerizing, serving at high throughput, observing drift & latency, and automating safe CI/CD rollouts.

---

## 📓 Notebook Curriculum

| # | Notebook | Domain | Key Concepts | Status |
|:--:|----------|--------|--------------|:------:|
| **01** | [`01-docker-for-ml.ipynb`](01-docker-for-ml.ipynb) | Containerization | Multi-stage Dockerfiles, Copy-on-Write layer caching, `.dockerignore`, Docker Compose stacks, NVIDIA GPU passthrough, CIS container security linting | ✅ Complete |
| **02** | [`02-model-serving-vllm.ipynb`](02-model-serving-vllm.ipynb) | LLM & Model Serving | PagedAttention KV block allocation, copy-on-write prefix sharing, continuous iteration-level batching, FastAPI OpenAI-compatible API, ONNX Runtime & load testing | ✅ Complete |
| **03** | [`03-monitoring-observability.ipynb`](03-monitoring-observability.ipynb) | Observability & Drift | Prometheus `Counter`/`Histogram`/`Gauge` registries, structured JSON logging, token/cost governance, PSI, Kolmogorov-Smirnov test, embedding centroid drift | ✅ Complete |
| **04** | [`04-ci-cd-pipelines.ipynb`](04-ci-cd-pipelines.ipynb) | CI/CD & Registry | GitHub Actions ML workflows, `pytest` behavioral invariance & latency tests, `nbformat` CI quality gates, SHA-256 Model Registry, Canary auto-rollback | ✅ Complete |

---

## 🗺️ Production MLOps Lifecycle

```
    [Notebook 04: CI/CD & Model Registry]
    - pytest behavioral & latency tests
    - nbformat execution gates
    - SHA-256 versioned artifacts
                     |
                     v
    [Notebook 01: Multi-Stage Docker & Compose]
    - Wheel builder -> non-root minimal runtime
    - Docker Compose (API + Redis + Prometheus)
    - NVIDIA Container Toolkit GPU reservations
                     |
                     v
    [Notebook 02: High-Throughput Serving]
    - PagedAttention KV-cache block manager
    - Continuous batching scheduler
    - FastAPI /v1/chat/completions & /health
                     |
                     v
    [Notebook 03: Monitoring & Observability]
    - Prometheus latency & TTFT histograms
    - Structured JSON logs + token burn alerts
    - PSI / KS / Embedding drift triggers -> Retrain / Rollback
```

---

## 🛠️ Packages Used

```python
fastapi >= 0.110.0
uvicorn >= 0.29.0
prometheus-client >= 0.20.0
pytest >= 8.0.0
httpx >= 0.27.0
pyyaml >= 6.0.0
pydantic >= 2.6.0
torch >= 2.0.0
onnxruntime >= 1.17.0
scipy >= 1.12.0
```

---

**[← Notebooks Index](../README.md)** · **[← Phase 4: Generative AI](../04-generative-ai/)** · **[Next: Phase 6: Advanced Paradigms →](../06-advanced-paradigms/)**
