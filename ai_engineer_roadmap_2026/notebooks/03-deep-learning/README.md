# 🧠 Phase 3: Deep Learning & Neural Networks (PyTorch & TensorFlow)

> **Duration**: 4–5 weeks · **Difficulty**: 🟠 Advanced · **Prerequisites**: Phase 1 (Foundations) & Phase 2 (ML Fundamentals)

Master representation learning and modern deep learning architectures across both industry-standard frameworks: **PyTorch** and **TensorFlow / Keras**. This phase bridges classical machine learning with foundation models, exploring the design philosophies, execution paradigms, performance tradeoffs, and production use cases of both ecosystems.

---

## 📓 Notebook Curriculum

| # | Notebook | Topic | Key Concepts | Status |
|:-:|----------|-------|--------------|:------:|
| 1 | [`01-framework-fundamentals-pytorch-vs-tensorflow.ipynb`](01-framework-fundamentals-pytorch-vs-tensorflow.ipynb) | **PyTorch vs. TensorFlow Fundamentals** | Dynamic DAG (Autograd) vs. `GradientTape` / `@tf.function`, `nn.Module` vs. `keras.Model`, data pipelines (`DataLoader` vs. `tf.data`), side-by-side training loop comparison, pros & cons | 📋 Planned |
| 2 | [`02-neural-networks-from-scratch.ipynb`](02-neural-networks-from-scratch.ipynb) | **Neural Networks from Scratch (NumPy)** | Pure NumPy matrix calculus backpropagation, activations (ReLU, GELU, Swish), Softmax + Cross-Entropy gradient derivation, gradient verification against both PyTorch Autograd & TensorFlow `GradientTape` | 📋 Planned |
| 3 | [`03-cnns-image-classification-pytorch-tf.ipynb`](03-cnns-image-classification-pytorch-tf.ipynb) | **CNNs, Vision Transfer Learning & Edge Deployment** | Convolutions, pooling, receptive fields, PyTorch `torchvision` vs. Keras vision models, transfer learning with ResNet, deployment benchmarks: TorchScript/ONNX vs. TensorFlow Lite (`.tflite`) | 📋 Planned |
| 4 | [`04-rnns-sequence-modeling-pytorch-tf.ipynb`](04-rnns-sequence-modeling-pytorch-tf.ipynb) | **Sequence Modeling with RNNs & LSTMs** | Recurrent unrolling, vanishing gradients, LSTM memory gates, GRUs, PyTorch `nn.LSTM` vs. Keras `layers.LSTM`, masking, time-series forecasting, streaming API latency tradeoffs | 📋 Planned |
| 5 | [`05-transformers-attention-pytorch-tf.ipynb`](05-transformers-attention-pytorch-tf.ipynb) | **Transformers & Multi-Head Attention** | Scaled Dot-Product Attention from scratch, Multi-Head Attention in PyTorch and TensorFlow, Positional Encodings (Sinusoidal & RoPE), Transformer Encoder block, generative LLM ecosystem adoption analysis | 📋 Planned |
| 6 | [`06-training-best-practices-pytorch-tf.ipynb`](06-training-best-practices-pytorch-tf.ipynb) | **Production Training, Acceleration & Deployment** | Cosine LR with warmup, AdamW vs. Adam, gradient clipping & accumulation, mixed precision (`torch.amp` vs. `tf.keras.mixed_precision`), XLA compilation, checkpointing, serving matrix (TorchServe, vLLM, TF Serving, TFLite) | 📋 Planned |

---

## ⚖️ Framework Tradeoffs: PyTorch vs. TensorFlow

```
┌────────────────────────────────────────────────────────┐
│                        PYTORCH                         │
│  • Leader in Generative AI, LLMs & Foundation Research │
│  • Dynamic Eager Autograd (Intuitive Pythonic logic)   │
│  • Ecosystem: Hugging Face, vLLM, DeepSpeed, FSDP      │
│  • Serving: TorchServe, Triton, TensorRT-LLM, ONNX     │
└────────────────────────────────────────────────────────┘
                           ▲
                           │ Core Mathematical Theory (NumPy)
                           ▼
┌────────────────────────────────────────────────────────┐
│                  TENSORFLOW / KERAS                    │
│  • Leader in Enterprise Serving & Industrial Pipelines │
│  • Static Graph Optimization via @tf.function & XLA    │
│  • Mobile & Edge: TensorFlow Lite (.tflite)            │
│  • In-Browser Inference: TensorFlow.js                 │
│  • Multi-Backend Abstraction via Keras 3               │
└────────────────────────────────────────────────────────┘
```

---

## 🛠️ Packages Used

```python
torch >= 2.0.0
torchvision >= 0.15.0
tensorflow >= 2.15.0
keras >= 3.0.0
numpy >= 1.24.0
scipy >= 1.10.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
```

---

**[← Notebooks Index](../README.md)** · **[← Phase 2: ML Fundamentals](../02-ml-fundamentals/)** · **[Next: Phase 4: Generative AI →](../04-generative-ai/)**
