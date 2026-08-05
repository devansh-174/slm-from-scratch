# Physics Small Language Model (Phase 1 - 25M)

A Physics-specific decoder-only Transformer (~25 million parameters) developed entirely from scratch without using any pretrained language models. This repository contains the complete implementation pipeline, including tokenizer analysis, tokenizer training, data preprocessing, model implementation, model training, and inference.

---

# Features

- Physics-specific decoder-only Transformer (~25M parameters)
- Built entirely from scratch without pretrained language models
- Custom SentencePiece tokenizer
- Complete tokenizer analysis, training, and verification pipeline
- Efficient binary dataset generation
- End-to-end model training with checkpointing
- Interactive Physics question answering through inference

---

# Repository Structure

```text
slm-from-scratch/
│
└── phase1_25m/
    │
    ├── README.md
    ├── .gitignore
    │
    ├── data/
    │   ├── sample_train.jsonl
    │   └── validation.jsonl
    │   
    ├── preprocessing/
    │   ├── make_bins.py
    │   └── make_bins.rmd
    │
    ├── tokenizer/
    │   ├── tokenizer_analysis.py
    │   ├── tokenizer_analysis.rmd
    │   ├── tokenizer_training.py
    │   ├── tokenizer_training.rmd
    │   ├── tokenizer_verify.py
    │   └── tokenizer_verify.rmd
    │
    ├── model/
    │   ├── model.py
    │   └── model.rmd
    │
    ├── train/
    │   ├── train.py
    │   └── train.rmd
    │
    ├── inference/
    │   ├── inference.py
    │   └── inference.rmd
    │
    └── outputs/
        ├── sample_validation.jsonl
        └── README.md
```

---

# Development Pipeline

```text
Dataset
   │
   ▼
Tokenizer Analysis
   │
   ▼
Tokenizer Training
   │
   ▼
Tokenizer Verification
   │
   ▼
Dataset Tokenization (.bin)
   │
   ▼
Model Training
   │
   ▼
Inference
```

---

# Components

## Data

Contains a sample validation dataset illustrating the expected dataset format.

---

## Preprocessing

Converts serialized JSONL datasets into binary token files for efficient model training.

| File | Description |
|------|-------------|
| `make_bins.py` | Converts serialized datasets into binary token files |
| `make_bins.rmd` | Documentation for the preprocessing stage |

---

## Tokenizer

Implements the complete SentencePiece tokenizer pipeline.

| File | Description |
|------|-------------|
| `tokenizer_analysis.py` | Analyzes the training corpus and evaluates tokenizer configurations |
| `tokenizer_training.py` | Trains the production SentencePiece tokenizer |
| `tokenizer_verify.py` | Independently verifies the trained tokenizer |

---

## Model

Implements the decoder-only Transformer architecture.

| File | Description |
|------|-------------|
| `model.py` | Transformer model implementation |
| `model.rmd` | Documentation of the model architecture |

---

## Training

Contains the complete model training pipeline.

| File | Description |
|------|-------------|
| `train.py` | Model training, validation, and checkpointing |
| `train.rmd` | Documentation for the training pipeline |

---

## Inference

Loads trained checkpoints and generates answers for Physics questions.

| File | Description |
|------|-------------|
| `inference.py` | Interactive inference script |
| `inference.rmd` | Documentation for the inference pipeline |

---

## Output

Contains sample validation outputs generated during model evaluation.

---

# Model Summary

| Specification | Value |
|--------------|-------|
| Domain | Physics |
| Architecture | Decoder-only Transformer |
| Parameters | ~25 Million |
| Tokenizer | SentencePiece Unigram |
| Positional Encoding | Rotary Positional Embedding (RoPE) |
| Normalization | RMSNorm |
| Feed Forward Network | SwiGLU |
| Context Length | 512 Tokens |
| Framework | PyTorch |

---

# Technologies

- Python
- PyTorch
- SentencePiece
- NumPy

---

# Training

Run the training pipeline:

```bash
python train/train.py
```

---

# Inference

Run interactive inference:

```bash
python inference/inference.py
```

---

# Sample Output

A sample validation output generated during model evaluation is available in:

```text
output/sample_validation.jsonl
```

---

# Note

This repository contains the implementation of the Phase 1 (25M) Physics Small Language Model. Large datasets, tokenizer artifacts, generated binary files (`.bin`), model checkpoints, and training logs are intentionally excluded to keep the repository lightweight and focused on the implementation.