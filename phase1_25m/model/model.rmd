# Model Architecture

## Overview

This stage implements the initial Physics Small Language Model (SLM), a decoder-only Transformer built entirely from scratch in PyTorch for autoregressive next-token prediction.

The architecture incorporates modern Transformer components including RoPE, RMSNorm, SwiGLU, Multi-Head Self Attention, and KV-cache based decoding.

---

## Architecture

| Component | Configuration |
|-----------|--------------|
| Model Type | Decoder-only Transformer |
| Parameters | ~26M |
| Vocabulary Size | 12,736 |
| Hidden Size | 512 |
| Layers | 6 |
| Attention Heads | 8 |
| Context Length | 512 |
| Positional Encoding | RoPE |
| Normalization | RMSNorm |
| Feed Forward | SwiGLU |
| Weight Tying | Enabled |

---

## Features

- Decoder-only Transformer
- Multi-Head Self Attention
- Rotary Positional Embeddings (RoPE)
- RMSNorm
- SwiGLU Feed Forward Network
- Fused QKV Projection
- Weight Tying
- KV-cache for fast inference
- GPT-style weight initialization
- Autoregressive text generation

---

## Workflow

```text
Input Tokens
      │
      ▼
Token Embeddings
      │
      ▼
Transformer Blocks × 6
      │
      ▼
Final RMSNorm
      │
      ▼
Language Head
      │
      ▼
Next Token Prediction
```

---

## Outcome

This stage defines the complete transformer architecture that serves as the foundation for model pretraining, inference, and subsequent fine-tuning.