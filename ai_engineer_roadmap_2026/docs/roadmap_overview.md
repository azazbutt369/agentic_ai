---
title: Roadmap Overview
description: A visual overview of the 6-phase AI Engineer learning path.
---

# 🗺️ Roadmap Overview

A structured, phase-by-phase learning path from absolute beginner to production AI Engineer.

**Estimated Total Duration:** 22–28 weeks (5–7 months) at 15–20 hours/week

---

## Visual Learning Path

```mermaid
graph TD
    START([🎯 START HERE]) --> P1
    
    subgraph P1 ["📐 PHASE 1: Foundations — 3-4 weeks"]
        direction LR
        P1A["🔢 Linear Algebra"] --> P1D["🐍 Python Mastery"]
        P1B["📈 Calculus"] --> P1D
        P1C["🎲 Probability"] --> P1D
        P1D --> P1E["📊 NumPy/Pandas/Polars"]
    end
    
    P1 --> P2
    
    subgraph P2 ["📊 PHASE 2: ML Fundamentals — 3-4 weeks"]
        direction LR
        P2A["📋 Supervised Learning"] --> P2D["🔧 scikit-learn"]
        P2B["🔍 Unsupervised Learning"] --> P2D
        P2C["📏 Evaluation Metrics"] --> P2D
        P2D --> P2E["⚙️ Feature Engineering"]
    end
    
    P2 --> P3
    
    subgraph P3 ["🧠 PHASE 3: Deep Learning — 4-5 weeks"]
        direction LR
        P3A["🔥 PyTorch"] --> P3B["🖼️ CNNs"]
        P3A --> P3C["📝 RNNs/LSTMs"]
        P3B --> P3D["⚡ Transformers"]
        P3C --> P3D
    end
    
    P3 --> P4
    
    subgraph P4 ["🤖 PHASE 4: GenAI & LLM Engineering — 6-8 weeks"]
        direction LR
        P4A["💬 Prompting"] --> P4B["🔗 RAG"]
        P4A --> P4C["🤖 Agents"]
        P4B --> P4D["🎯 Advanced RAG"]
        P4C --> P4E["👥 Multi-Agent"]
        P4D --> P4F["🔧 Fine-Tuning"]
        P4E --> P4F
        P4F --> P4G["⚖️ DPO"]
    end
    
    P4 --> P5
    
    subgraph P5 ["⚙️ PHASE 5: MLOps & LLMOps — 3-4 weeks"]
        direction LR
        P5A["🐳 Docker"] --> P5B["🚀 Model Serving"]
        P5B --> P5C["📊 Monitoring"]
        P5C --> P5D["🔄 CI/CD"]
    end
    
    P5 --> P6
    
    subgraph P6 ["🔮 PHASE 6: Advanced Paradigms — 2-3 weeks"]
        direction LR
        P6A["🎭 Multimodal AI"]
        P6B["📱 Edge AI & SLMs"]
        P6C["🛡️ AI Safety"]
    end
    
    P6 --> FINISH([🏆 AI ENGINEER])
```

---

## Phase Summary

### Difficulty Legend

| Symbol | Level | Description |
|:------:|-------|-------------|
| 🟢 | Beginner | No prior knowledge required |
| 🟡 | Intermediate | Requires foundational understanding |
| 🟠 | Advanced | Requires solid intermediate skills |
| 🔴 | Expert | Production-level, cutting-edge topics |

### Phase Details

| Phase | Duration | Notebooks | Difficulty | Key Outcomes |
|-------|----------|:---------:|:----------:|--------------|
| [1. Foundations](phases/01-foundations.md) | 3–4 wks | 7 | 🟢 | Math fluency, Python mastery, data manipulation |
| [2. ML Fundamentals](phases/02-ml-fundamentals.md) | 3–4 wks | 6 | 🟡 | Classical ML, scikit-learn, feature engineering |
| [3. Deep Learning](phases/03-deep-learning.md) | 4–5 wks | 6 | 🟠 | PyTorch, CNNs, Transformers from scratch |
| [4. Generative AI](phases/04-generative-ai.md) | 6–8 wks | 9 | 🟠–🔴 | LLMs, RAG, agents, fine-tuning, DPO |
| [5. MLOps](phases/05-mlops-llmops.md) | 3–4 wks | 4 | 🟠 | Docker, vLLM, monitoring, CI/CD |
| [6. Advanced](phases/06-advanced-paradigms.md) | 2–3 wks | 3 | 🔴 | Multimodal, edge AI, safety |
| **Total** | **22–28 wks** | **35** | | |

---

## How to Use This Roadmap

1. **Follow the phases in order** — each phase builds on the previous one
2. **Complete the notebooks** before moving to the next phase
3. **Build at least one project** from Phases 4–6 for your portfolio
4. **Check off completed items** to track your progress
5. **Don't rush** — deep understanding beats surface-level coverage

!!! tip "Skip Ahead?"
    If you already have experience in certain areas, you can skim the review sections. But we strongly recommend at least skimming Phase 3 (especially the Transformer notebook) before jumping into Phase 4.

---

## What Comes After?

After completing all 6 phases, you will have:

- ✅ Deep understanding of mathematics, ML, and deep learning
- ✅ Mastery of the Transformer architecture
- ✅ Hands-on experience with LLMs, RAG, and AI agents
- ✅ Fine-tuning and alignment skills (LoRA, DPO)
- ✅ Production deployment with Docker and vLLM
- ✅ Knowledge of frontier topics (multimodal, edge AI, safety)
- ✅ 7 portfolio-ready projects
- ✅ The skills to pass AI Engineer interviews at top companies
