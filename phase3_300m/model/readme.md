## Model Architecture

PyTorch implementation of a physics-domain, decoder-only Transformer with approximately **300M Total parameters**.

| Component | Configuration |
|---|---:|
| Total Parameters | 300,043,776 |
| Vocabulary size | 11,000 |
| Hidden size | 1,536 |
| Transformer layers | 10 |
| Query / KV heads | 24 / 6 |
| Head dimension | 64 |
| Maximum context length | 2,048 tokens |
| FFN intermediate size | 4,864 |
| Attention | Grouped-query attention (GQA) |
| Positional encoding | RoPE |
| Feed-forward network | SwiGLU |
| Normalization | RMSNorm, Pre-LN |
| Embedding/output weights | Tied |
| Attention implementation | PyTorch SDPA |

