# 🧠 Phase 3: Deep Learning & Neural Networks

> **Duration**: 4–5 weeks · **Difficulty**: 🟠 Advanced · **Prerequisites**: Phase 1 (Foundations) & Phase 2 (ML Fundamentals)

Master representation learning and modern neural network architectures in **PyTorch**. This phase bridges classical machine learning with modern foundation models, covering everything from tensor computation graphs and backpropagation from scratch to deep CNNs, sequential RNNs/LSTMs, and the multi-head self-attention mechanism powering Transformers.

---

## 📓 Notebook Curriculum

| # | Notebook | Topic | Key Concepts | Status |
|:-:|----------|-------|--------------|:------:|
| 1 | [`01-pytorch-fundamentals.ipynb`](01-pytorch-fundamentals.ipynb) | **PyTorch Fundamentals** | Tensors, Autograd graphs, `.backward()`, `nn.Module`, optimizers, GPU/device handling, custom datasets & `DataLoader` | 📋 Planned |
| 2 | [`02-neural-networks-from-scratch.ipynb`](02-neural-networks-from-scratch.ipynb) | **Neural Networks from Scratch** | Multi-Layer Perceptrons (MLPs), manual backpropagation in pure NumPy, activations (ReLU, GELU, Swish), Softmax + Cross-Entropy gradient derivation | 📋 Planned |
| 3 | [`03-cnns-image-classification.ipynb`](03-cnns-image-classification.ipynb) | **Convolutional Neural Networks (CNNs)** | 2D Convolutions, pooling, receptive fields, residual connections (ResNet), transfer learning, torchvision augmentations | 📋 Planned |
| 4 | [`04-rnns-sequence-modeling.ipynb`](04-rnns-sequence-modeling.ipynb) | **Sequence Modeling with RNNs & LSTMs** | Recurrent unrolling, vanishing/exploding gradients, LSTM gating mechanisms, GRUs, sequential forecasting & classification | 📋 Planned |
| 5 | [`05-transformers-attention.ipynb`](05-transformers-attention.ipynb) | **Transformers & Self-Attention** | Scaled Dot-Product Attention from scratch, Multi-Head Attention (MHA), Positional Encodings (Sinusoidal & RoPE), Transformer Encoder block | 📋 Planned |
| 6 | [`06-training-best-practices.ipynb`](06-training-best-practices.ipynb) | **Deep Learning Training Best Practices** | Cosine LR scheduling with warmup, AdamW & weight decay, gradient clipping, gradient accumulation, mixed precision (`fp16`/`bf16`), checkpointing | 📋 Planned |

---

## 🗺️ Learning Path

```
                    ┌──▶ 03. CNNs & Vision ────────┐
01. PyTorch ──▶ 02. MLP from Scratch               ├──▶ 05. Transformers ──▶ 06. Training Best Practices
                    └──▶ 04. RNNs & Sequences ─────┘
```

1. **Foundations**: Master PyTorch tensors and autograd (`01`), then demystify the magic by coding forward/backward passes in pure NumPy (`02`).
2. **Specialized Architectures**: Learn spatial feature extraction via CNNs (`03`) and temporal state tracking via RNNs/LSTMs (`04`).
3. **Modern Era**: Construct the Transformer Self-Attention engine from scratch (`05`), then harden models with production training techniques (`06`).

---

## 🛠️ Tech Stack & Packages

```python
torch >= 2.0.0
torchvision >= 0.15.0
numpy >= 1.24.0
scipy >= 1.10.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
```

---

## 📐 10-Section Notebook Standard

Every notebook in Phase 3 adheres strictly to the curriculum quality guarantees:
1. **Overview** — Architectural intuition and industry relevance.
2. **Learning Objectives** — Actionable, measurable engineering goals.
3. **Imports** — Pinned library versions with runtime verification.
4. **Configuration** — Explicit random seeds and device management (`cpu` / `cuda` / `mps`).
5. **Theory & Mathematical Foundations** — Rigorous LaTeX formulas, forward/backward derivations, and architectural diagrams.
6. **Implementation** — Modular, type-annotated PyTorch/Python code.
7. **Evaluation & Diagnostics** — Loss curves, gradient norms, attention heatmaps, and confusion matrices.
8. **Guided Exercises** — 2 structured practice problems with complete verified solutions.
9. **Challenge Problems** — 2 advanced, portfolio-worthy engineering challenges.
10. **Further Reading** — Seminal research papers, PyTorch documentation, and reference books.

---

**[← Notebooks Index](../README.md)** · **[← Phase 2: ML Fundamentals](../02-ml-fundamentals/)** · **[Next: Phase 4: Generative AI →](../04-generative-ai/)**
