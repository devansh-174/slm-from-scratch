# Production Tokenizer Training

## Overview

This stage trains the final production SentencePiece tokenizer using the configuration selected during the tokenizer analysis stage.

The generated tokenizer is used consistently throughout dataset preparation, model pretraining, inference, and fine-tuning.

---

## Features

- Corpus serialization
- Placeholder cleaning
- SentencePiece Unigram training
- Automatic vocabulary adjustment
- Custom special token support (`<CALC>`, `</CALC>`)
- Encode/decode validation
- Production tokenizer generation

---

## Workflow

```text
Training Dataset
      │
      ▼
Corpus Serialization
      │
      ▼
Placeholder Cleaning
      │
      ▼
SentencePiece Training
      │
      ▼
Tokenizer Validation
      │
      ▼
Production Tokenizer
```

---

## Outputs

- `slm_tokenizer.model`
- `slm_tokenizer.vocab`
- `train_corpus.txt`

---

## Outcome

Generates the final production tokenizer used across all subsequent stages of the Physics Small Language Model pipeline.