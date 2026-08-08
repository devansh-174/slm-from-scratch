# `model1.py`

Decoder-only Transformer for the **Phase 2 Physics SLM (~25M parameters)**.

## Configuration

| Component | Value |
|---|---|
| Model type | Decoder-only Transformer |
| Vocabulary | 11,000 |
| Hidden size | 512 |
| Layers | 6 |
| Attention heads | 8 |
| Head dimension | 64 |
| Context length | 1,400 |
| FFN | SwiGLU |
| FFN intermediate size | 1,408 |
| Positional encoding | RoPE |
| RoPE θ | 10,000 |
| Normalization | RMSNorm |
| Architecture | Pre-LN |
| Attention | MHA |
| QKV projection | Fused |
| Dropout | 0.075 |
| Attention dropout | 0.0 |
| Weight tying | Enabled |
| Linear bias | Disabled |