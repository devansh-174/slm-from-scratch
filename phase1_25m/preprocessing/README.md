# Dataset Binarization

## Overview

This stage converts the prepared JSONL dataset into binary token files (`.bin`) required for efficient model pretraining.

The script serializes each training sample using the same format employed during tokenizer training, tokenizes the serialized text using the trained SentencePiece tokenizer, and stores the resulting token IDs as compact binary arrays.

---

## Objectives

- Load the training dataset.
- Maintain the same serialization format used during tokenizer training.
- Tokenize every training sample using the trained SentencePiece tokenizer.
- Generate binary token files for efficient model training.
- Automatically create a validation split when one is unavailable.
- Prevent data leakage by splitting on `source_concept_id`.

---

## Features

- JSONL dataset loading
- Consistent dataset serialization
- SentencePiece tokenization
- Automatic train/validation split
- Concept-level data splitting
- EOS token insertion between samples
- Binary dataset generation (`.bin`)
- Memory-efficient storage using `uint16`

---

## Workflow

```text
JSONL Dataset
      │
      ▼
Dataset Serialization
      │
      ▼
SentencePiece Tokenization
      │
      ▼
Train / Validation Split
      │
      ▼
EOS Token Insertion
      │
      ▼
Binary Token Files (.bin)
```

---

## Outputs

The script generates:

- `train.bin`
- `val.bin`

These binary files are used directly by the pretraining pipeline.

---

## Outcome

This stage transforms the cleaned dataset into an optimized binary representation, enabling efficient loading and high-throughput training during the pretraining stage.
