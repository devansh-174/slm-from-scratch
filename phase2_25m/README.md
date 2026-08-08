# Physics Small Language Model — Phase 2 (~25M)

A physics-domain Small Language Model (SLM) developed from scratch using a decoder-only Transformer architecture.

## Project Structure

```text
phase2_25m/
├── data/
├── tokenizer/
├── preprocessing/
├── model/
├── train/
├── inference/
└── README.md
```

## Pipeline

```text
Training Data
      ↓
Tokenizer Training
      ↓
Tokenizer Verification
      ↓
Binary Preprocessing
      ↓
Model Training
      ↓
Checkpoints
      ↓
Inference
```

## Model

| Component | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | ~25M |
| Vocabulary | 11,000 |
| Hidden size | 512 |
| Layers | 6 |
| Attention heads | 8 |
| Head dimension | 64 |
| Context length | 1,400 |
| Attention | Multi-Head Self-Attention |
| QKV | Fused |
| Positional encoding | RoPE |
| FFN | SwiGLU |
| FFN intermediate size | 1,408 |
| Normalization | RMSNorm |
| Architecture | Pre-LN |
| Weight tying | Enabled |
| Linear bias | Disabled |

## Tokenizer

SentencePiece Unigram tokenizer.

| Setting | Value |
|---|---|
| Vocabulary | 11,000 |
| Normalization | NFKC |
| Character coverage | 1.0 |
| Lowercasing | OFF |
| Byte fallback | ON |
| Digit splitting | ON |
| Custom tokens | None |
| UNK / BOS / EOS / PAD | 0 / 1 / 2 / 3 |
| Vocabulary padding | OFF |
| Placeholder cleaning | ON |

## Data Preprocessing

`make_bins1.py` converts the JSONL datasets into binary token-ID files using the production tokenizer.

**Input**
- `train1.jsonl`
- `validation1.jsonl`

**Output**
- `train1.bin`
- `val1.bin`

Binary token IDs are stored as `uint16`.

## Training

The model is trained from scratch using the generated binary token-ID datasets.

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| Weight decay | 0.01 |
| Scheduler | Linear warmup → Cosine decay |
| Warmup | 3% |
| Precision | FP32 |
| Context length | 1,400 |
| Micro batch size | 8 |
| Gradient accumulation | 4 |
| Effective batch size | 32 |
| Gradient clipping | 1.0 |
| Maximum epochs | 100 |
| Validation interval | 500 steps |
| Checkpoint interval | 500 steps |
| Early stopping | 15 validations |
| GPUs | 2 × Quadro P5000 |
| Seed | 42 |

## Checkpoints

```text
checkpoints/
├── best_model.pt
├── latest_model.pt
└── final_model.pt
```

- `best_model.pt` — best validation loss
- `latest_model.pt` — latest resumable checkpoint
- `final_model.pt` — final checkpoint

## Inference

Run from the project root:

```bash
python3 inference/inference1.py
```

## Components

- `tokenizer/` — tokenizer training and verification
- `preprocessing/` — JSONL to binary preprocessing
- `model/` — model architecture
- `train/` — model training and checkpointing
- `inference/` — model inference and generation

## Execution Order

1. `tokenizer_train.py`
2. `tokenizer_verify.py`
3. `make_bins1.py`
4. `train.py`
5. `inference1.py`

The tokenizer should be successfully verified before generating the binary training data and starting model training.