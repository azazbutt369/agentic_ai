---
title: Glossary
description: Definitions for AI/ML terms used throughout the AI Engineer Roadmap 2026.
---

# 📖 Glossary

Key terms and concepts used throughout the AI Engineer Roadmap 2026, organized alphabetically.

---

## A

**Activation Function**
:   A mathematical function applied to a neuron's output to introduce non-linearity. Common examples: ReLU, GELU, Sigmoid, Softmax. Without activation functions, neural networks would only learn linear relationships.

**Agent (AI Agent)**
:   An AI system that can autonomously perceive its environment, make decisions, and take actions to achieve goals. Modern AI agents use LLMs for reasoning and can call external tools (search, code execution, APIs).

**Attention Mechanism**
:   A technique that allows models to focus on different parts of the input when generating each part of the output. Self-attention is the core innovation behind Transformers.

**Autograd**
:   PyTorch's automatic differentiation engine. It records operations on tensors and automatically computes gradients during backpropagation.

**AWQ (Activation-aware Weight Quantization)**
:   A quantization method that reduces model size by observing which weights are most important based on activation patterns, preserving accuracy better than naive quantization.

---

## B

**Backpropagation**
:   The algorithm used to train neural networks by computing gradients of the loss function with respect to each weight, working backward from the output layer.

**Bayes' Theorem**
:   A formula for updating probabilities based on new evidence: P(A|B) = P(B|A) × P(A) / P(B). Fundamental to probabilistic ML and language model sampling.

**BERT (Bidirectional Encoder Representations from Transformers)**
:   A Transformer-based model (encoder-only) designed for understanding text. Pre-trained on masked language modeling. Basis for many embedding models.

---

## C

**Chain-of-Thought (CoT)**
:   A prompting technique that asks the model to "think step by step," improving reasoning performance on complex tasks. Introduced in "Chain-of-Thought Prompting Elicits Reasoning" (Wei et al., 2022).

**Chunking**
:   In RAG, the process of splitting documents into smaller segments for embedding and retrieval. Strategies include fixed-size, recursive, and semantic chunking.

**CNN (Convolutional Neural Network)**
:   A neural network architecture specialized for processing grid-like data (images, time series). Uses convolutional layers to learn spatial features hierarchically.

**Cross-Entropy Loss**
:   A loss function commonly used for classification tasks. Measures the difference between predicted probability distribution and the true distribution.

---

## D

**DPO (Direct Preference Optimization)**
:   A method for aligning LLMs to human preferences without requiring a separate reward model. Simpler and more stable than RLHF. The 2026 standard for alignment.

**Dropout**
:   A regularization technique that randomly "drops" (sets to zero) a fraction of neurons during training to prevent overfitting.

---

## E

**Embedding**
:   A dense vector representation of data (text, images, etc.) in a continuous space where semantic similarity corresponds to vector proximity. Text embeddings power semantic search and RAG systems.

**Encoder-Decoder**
:   A Transformer architecture with an encoder (processes input) and decoder (generates output). Used in translation and sequence-to-sequence tasks. GPT is decoder-only; BERT is encoder-only.

---

## F

**Few-Shot Prompting**
:   A prompting technique where you provide the model with a few examples of the desired input-output behavior before asking it to perform a task.

**Fine-Tuning**
:   The process of training a pre-trained model on a specific dataset to adapt it for a particular task or domain. See also: LoRA, QLoRA, PEFT.

---

## G

**GGUF (GPT-Generated Unified Format)**
:   A file format for storing quantized LLMs, designed for efficient loading and inference. Used by llama.cpp and Ollama.

**Gradient Descent**
:   An optimization algorithm that iteratively adjusts model parameters in the direction that reduces the loss function. Variants: SGD, Adam, AdamW.

**GraphRAG**
:   An advanced RAG pattern that combines knowledge graphs with vector search for better retrieval of complex, relational information.

---

## H

**Hyperparameter**
:   A parameter set before training begins (learning rate, batch size, number of layers), as opposed to parameters learned during training (weights, biases).

---

## L

**LangChain**
:   A popular framework for building LLM-powered applications. Provides abstractions for chains, agents, retrieval, and memory.

**LangGraph**
:   A library for building stateful, multi-actor agent workflows as graphs. Built on top of LangChain.

**LoRA (Low-Rank Adaptation)**
:   A parameter-efficient fine-tuning method that freezes the pre-trained model and adds small trainable matrices to specific layers. Dramatically reduces memory and compute requirements.

**Loss Function**
:   A function that measures how far a model's predictions are from the true values. Training minimizes this function. Common types: MSE (regression), Cross-Entropy (classification).

**LSTM (Long Short-Term Memory)**
:   A type of RNN designed to learn long-term dependencies using gates (forget, input, output). Largely superseded by Transformers for most tasks.

---

## M

**MLP (Multi-Layer Perceptron)**
:   The simplest type of feedforward neural network, consisting of an input layer, one or more hidden layers, and an output layer.

**Multi-Head Attention**
:   An extension of the attention mechanism that runs multiple attention operations in parallel, each learning different types of relationships in the data.

---

## O

**Ollama**
:   An open-source tool for running LLMs locally on your machine. Supports Llama, Mistral, Phi, and many other models. Used throughout this curriculum as the primary LLM interface.

---

## P

**PEFT (Parameter-Efficient Fine-Tuning)**
:   A family of techniques for fine-tuning large models by training only a small number of additional parameters. Includes LoRA, QLoRA, prefix tuning, and adapters.

**Positional Encoding**
:   A mechanism for injecting sequence position information into Transformer models, which have no inherent notion of order. Can be sinusoidal (fixed) or learned.

**Prompt Engineering**
:   The practice of designing input prompts to guide LLM behavior. Techniques include zero-shot, few-shot, chain-of-thought, and system prompts.

---

## Q

**QLoRA (Quantized LoRA)**
:   A combination of 4-bit quantization and LoRA that enables fine-tuning of large models on consumer GPUs (8GB+ VRAM).

**Quantization**
:   Reducing model precision (e.g., from 32-bit float to 4-bit integer) to decrease memory usage and increase inference speed. Methods: GGUF, AWQ, EXL2.

---

## R

**RAG (Retrieval-Augmented Generation)**
:   A pattern that enhances LLM responses by first retrieving relevant documents from a knowledge base, then including them as context in the prompt. Reduces hallucination and allows knowledge updates without retraining.

**ReAct (Reasoning and Acting)**
:   A prompting framework where the model alternates between reasoning (thinking about what to do) and acting (calling tools or taking actions). The basis for most modern AI agents.

**RNN (Recurrent Neural Network)**
:   A neural network architecture designed for sequential data, where the output from the previous step is fed as input to the current step.

---

## S

**Self-Attention**
:   A mechanism where each element in a sequence attends to every other element, computing relevance scores. The core operation in Transformer models.

**Semantic Search**
:   Search based on meaning rather than keyword matching. Uses embeddings to find semantically similar content in a vector database.

---

## T

**Tensor**
:   A multi-dimensional array. Scalars (0D), vectors (1D), matrices (2D), and higher-dimensional arrays are all tensors. The fundamental data structure in deep learning frameworks.

**Transformer**
:   The dominant neural network architecture (since 2017). Based on self-attention mechanisms. Powers GPT, BERT, Llama, Mistral, and virtually all modern AI models. Introduced in "Attention Is All You Need" (Vaswani et al., 2017).

---

## V

**Vector Database**
:   A database optimized for storing and querying high-dimensional vectors (embeddings). Used in RAG systems for semantic search. Examples: Chroma, Qdrant, Milvus, Weaviate.

**vLLM**
:   A high-throughput LLM serving engine that uses PagedAttention for efficient memory management and continuous batching for maximizing GPU utilization.
