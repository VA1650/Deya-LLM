# 🧠 DeiaGPT: Custom Language Model Research

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An architectural deep dive into training custom autoregressive Transformers using FlashAttention, Rotary Position Embeddings (RoPE), and deep memory optimization techniques.

---

## 📊 Architectural Evolution: V9 vs V10

This project documents a classic LLM engineering trade-off: scaling model capacity vs. maintaining convergence quality under strict hardware constraints.

| Feature / Metric | Version 9 (Stable Baseline) | Version 10 (Experimental Large) |
| :--- | :--- | :--- |
| **Model Size** | ~130M+ parameters ($d_{model}=768$, 32 layers) | ~300M+ parameters ($d_{model}=1024$, 32 layers) |
| **Attention** | Native PyTorch SDPA (FlashAttention enabled) | Optimized SDPA via explicit RoPE indexing |
| **Optimizer** | FP32 `AdamW` (High fidelity) | `AdamW8bit` (BitsAndBytes memory-optimized) |
| **LR Scheduling** | Constant ($1e-4$) | Cosine Decay with Warmup ($3e-4 \to 3e-5$) |
| **Batch Config** | Effective Batch: 32 ($4 \times 8$) | Effective Batch: 64 ($1 \times 64$) |
| **Output Quality** | 🏆 **High coherence, stable text generation** | Lower text quality (potential underfitting) |

---

## 🔬 Critical Post-Mortem & Engineering Insights

### 1. The V9 VRAM Leak / Slowdown Bug
* **The Problem:** Resuming training from checkpoint in V9 caused a 10x performance drop and memory bloating.
* **The Root Cause:** Standard FP32 `AdamW` allocates internal momentum tensors equal to $2 \times$ model parameters. Reloading the state dict caused severe CUDA memory fragmentation on the GPU, breaking subsequent memory allocations.

### 2. Why V10 Underperformed Despite Superior Architecture
* **Parameter/Data Mismatch:** Doubling the embedding size ($768 \to 1024$) vastly increased the model's capacity. For small to mid-sized text datasets (`input.txt`), the model required significantly more training tokens to properly align its latent space.
* **8-bit Quantization Trade-off:** While `AdamW8bit` successfully resolved the V9 memory fragmentation bug and allowed training a larger model on a single GPU, the quantization noise in gradient updates slightly degraded fine-grained token predictions.
* **Batch Size Variance:** Dropping the micro-batch size to $1$ (with 64 accumulation steps) increased gradient variance during backward passes, creating sub-optimal update steps when combined with 8-bit optimizer states.

---

## 🛠 Features Implemented
* **RoPE (Rotary Position Embeddings):** Replaced absolute positional encodings with relative rotary embeddings for enhanced context extrapolation.
* **Memory Management:** Integrated PyTorch Gradient Checkpointing (`use_reentrant=False`) across all 32 layers to enable deep architecture training on consumer hardware.
* **Stability Enhancements:** Pre-LayerNorm architecture using `RMSNorm` instead of standard `LayerNorm` to prevent gradient explosion.