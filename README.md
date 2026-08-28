# 🧠 DeiaGPT: Custom Language Model Research

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An architectural deep dive into training custom autoregressive Transformers using FlashAttention, Rotary Position Embeddings (RoPE), and deep memory optimization techniques.

---

## 📊 Architectural Evolution: V9 vs V9.1 vs V10

This project documents a classic LLM engineering trade-off: trading off sequence modeling depth against tokenizer efficiency, inference throughput, and dataset markup structures.

| Feature / Metric | Version 9 (Stable Baseline) | Version 9.1 (Custom BPE & Speed) | Version 10 (Experimental Large) |
| :--- | :--- | :--- | :--- |
| **Tokenizer** | Standard GPT-2 BPE | **Custom Trained BPE** | Standard GPT-2 BPE |
| **Inference Speed** | Slow (~3 min / 1k tokens) | **High Throughput (~20-30s / 1k tokens)** | Moderate |
| **Dataset & Context** | Standard turns, multi-topic bleed | **Longer dialogues, refined stop tokens** | Scaled capacity |
| **Output Quality** | Stable, mixed topic drift | **Sharper topical focus, but weaker global semantics** | Lower text quality |
| **Checkpoints** | Bloated state dicts | **Lean, storage-optimized saves** | Bloated state dicts |

---

## 🔬 Critical Post-Mortem & Engineering Insights

### 1. Version 9.1: The Custom BPE & Inference Breakthrough
* **The Wins:** Replacing the generic GPT-2 tokenizer with a custom BPE implementation dramatically accelerated generation speeds—slashing inference time from ~3 minutes down to 20–30 seconds for 1k tokens. Combined with lean checkpointing logic, disk bloat and state-loading friction were effectively eliminated.
* **The Trade-Offs:** Although training loss dropped and topical focus improved compared to the multi-topic soup of V9 and V10, long-range dependency tracking remains a bottleneck. Because the dataset was refactored with longer dialogue turns, outputs are noticeably more verbose, yet overall semantic coherence trails behind the V9 baseline.

### 2. The V9 VRAM Leak / Slowdown Bug
* **The Problem:** Resuming training from checkpoint in V9 caused a 10x performance drop and memory bloating.
* **The Root Cause:** Standard FP32 `AdamW` allocates internal momentum tensors equal to $2 \times$ model parameters. Reloading the state dict caused severe CUDA memory fragmentation on the GPU, breaking subsequent memory allocations.

### 3. Why V10 Underperformed Despite Superior Architecture
* **Parameter/Data Mismatch:** Doubling the embedding size ($768 \to 1024$) vastly increased the model's capacity. For small to mid-sized text datasets (`input.txt`), the model required significantly more training tokens to properly align its latent space.
* **8-bit Quantization Trade-off:** While `AdamW8bit` successfully resolved the V9 memory fragmentation bug and allowed training a larger model on a single GPU, the quantization noise in gradient updates slightly degraded fine-grained token predictions.

---

## 🛠 Features Implemented
* **Custom BPE Tokenizer:** Domain-tailored vocabulary mapping optimized for fast decoding loops and dense token representation.
* **RoPE (Rotary Position Embeddings):** Replaced absolute positional encodings with relative rotary embeddings for enhanced context extrapolation.
* **Memory Management:** Integrated PyTorch Gradient Checkpointing (`use_reentrant=False`) across layers to enable deep architecture training on consumer hardware.
* **Stability Enhancements:** Pre-LayerNorm architecture using `RMSNorm` instead of standard `LayerNorm` to prevent gradient explosion.