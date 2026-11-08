# 🧠 Phase 3: Deep Learning & Neural Networks (PyTorch & TensorFlow)

> **Duration**: 4–5 weeks · **Difficulty**: 🟠 Advanced · **Prerequisites**: Phase 1 (Foundations) & Phase 2 (ML Fundamentals)

Master representation learning and modern deep learning architectures across **both** industry-standard frameworks: **PyTorch** and **TensorFlow / Keras**. This phase features **separate, dedicated notebooks** for each framework across every domain, providing specialized mastery of both research workflows and enterprise edge/serving deployment.

---

## 📓 Notebook Curriculum

| # | Notebook | Framework | Domain | Key Concepts | Status |
|:-:|----------|:---------:|--------|--------------|:------:|
| **01** | [`01-pytorch-fundamentals.ipynb`](01-pytorch-fundamentals.ipynb) | **PyTorch** | Fundamentals | Tensors, dynamic Autograd graphs, `.backward()`, `nn.Module`, optimizers, custom training loops, `DataLoader` | 📋 Planned |
| **02** | [`02-tensorflow-fundamentals.ipynb`](02-tensorflow-fundamentals.ipynb) | **TensorFlow** | Fundamentals | Tensors, `tf.Variable`, `GradientTape`, `@tf.function` & XLA graph compilation, Keras APIs, `tf.data` | 📋 Planned |
| **03** | [`03-neural-networks-from-scratch.ipynb`](03-neural-networks-from-scratch.ipynb) | **NumPy** | Math Foundations | Deep MLP from scratch in pure NumPy, manual matrix calculus backpropagation, dual gradient verification against PyTorch & TF | 📋 Planned |
| **04** | [`04-cnns-computer-vision-pytorch.ipynb`](04-cnns-computer-vision-pytorch.ipynb) | **PyTorch** | Computer Vision | 2D Convolutions, pooling, ResNet transfer learning, progressive unfreezing, Grad-CAM, ONNX export | 📋 Planned |
| **05** | [`05-cnns-computer-vision-tensorflow.ipynb`](05-cnns-computer-vision-tensorflow.ipynb) | **TensorFlow** | Vision & Edge | Keras Conv2D, MobileNetV2 transfer learning, **TensorFlow Lite (`.tflite`) conversion & INT8 quantization** | 📋 Planned |
| **06** | [`06-rnns-sequence-modeling-pytorch.ipynb`](06-rnns-sequence-modeling-pytorch.ipynb) | **PyTorch** | Sequences | Recurrent unrolling, vanishing gradients, `nn.LSTM` / `nn.GRU`, hidden states, packed padded sequences, time series | 📋 Planned |
| **07** | [`07-rnns-sequence-modeling-tensorflow.ipynb`](07-rnns-sequence-modeling-tensorflow.ipynb) | **TensorFlow** | Sequences | Keras `LSTM`, `GRU`, `Masking` layers for variable length, bidirectional models, streaming sequence inference | 📋 Planned |
| **08** | [`08-transformers-attention-pytorch.ipynb`](08-transformers-attention-pytorch.ipynb) | **PyTorch** | Transformers | Scaled Dot-Product Attention from scratch, Multi-Head Attention (MHA), RoPE, full Transformer Encoder, causal masking | 📋 Planned |
| **09** | [`09-transformers-attention-tensorflow.ipynb`](09-transformers-attention-tensorflow.ipynb) | **TensorFlow** | Transformers | Keras `MultiHeadAttention` layer, building Transformer Blocks in Keras, positional embeddings, KerasHub models | 📋 Planned |
| **10** | [`10-training-deployment-pytorch.ipynb`](10-training-deployment-pytorch.ipynb) | **PyTorch** | Production | Cosine LR with Warmup, AdamW, gradient clipping & accumulation, mixed precision (`torch.amp`), checkpointing, TorchServe | 📋 Planned |
| **11** | [`11-training-deployment-tensorflow.ipynb`](11-training-deployment-tensorflow.ipynb) | **TensorFlow** | Production | `@tf.function(jit_compile=True)` XLA compiler, `CosineDecay`, mixed precision, SavedModel, TF Serving (gRPC/REST) & TF.js | 📋 Planned |

---

## 🗺️ Learning Path Options

```
                    ┌──▶ [PyTorch Track] ────▶ 01 ──▶ 04 (Vision) ──▶ 06 (RNN) ──▶ 08 (Transformer) ──▶ 10 (Serving)
03. Math (NumPy) ───┤
                    └──▶ [TensorFlow Track] ─▶ 02 ──▶ 05 (TFLite) ──▶ 07 (RNN) ──▶ 09 (Transformer) ──▶ 11 (Serving)
```

- **Comparative Path (Recommended)**: Learn each concept pair side-by-side (`01 & 02`, `04 & 05`, `06 & 07`, `08 & 09`, `10 & 11`) to master framework tradeoffs.
- **Track-Specific Path**: Complete the entire PyTorch track for generative AI/LLM specialization, or the TensorFlow track for enterprise serving/edge specialization.

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
