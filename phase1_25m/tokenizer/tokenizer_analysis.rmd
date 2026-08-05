# Tokenizer Analysis

## Overview

This stage performs corpus analysis and trains the tokenizer for the initial Physics Small Language Model using SentencePiece. The script analyzes the training corpus, evaluates multiple vocabulary sizes, and recommends an initial tokenizer configuration.

---

## Objectives

- Prepare the training corpus for tokenizer training.
- Analyze corpus statistics and vocabulary distribution.
- Inspect Unicode characters and mathematical symbols.
- Detect unexpected or noisy characters.
- Train and compare multiple SentencePiece tokenizers.
- Estimate an appropriate context length.
- Recommend an initial tokenizer configuration.

---

## Features

- Dataset serialization
- Corpus statistics
- Unicode & symbol analysis
- Stray character detection
- Case sensitivity analysis
- Vocabulary size sweep
- SentencePiece tokenizer training
- Tokenizer evaluation
- Context length estimation
- Configuration recommendation

---

## Workflow

```text
Dataset
   │
   ▼
Corpus Serialization
   │
   ▼
Corpus Analysis
   │
   ▼
Unicode & Symbol Analysis
   │
   ▼
Vocabulary Sweep
   │
   ▼
SentencePiece Training
   │
   ▼
Tokenizer Evaluation
   │
   ▼
Configuration Recommendation
```

---

## Outputs

The script generates:

- Serialized training corpus (`corpus.txt`)
- SentencePiece tokenizer models (`spm_*.model`)
- Vocabulary files (`spm_*.vocab`)
- Corpus analysis reports
- Tokenizer evaluation statistics
- Flagged Unicode samples (`flagged_ids.txt`)
- Final tokenizer recommendation

---

## Outcome

This stage establishes a corpus-driven tokenizer configuration that serves as the foundation for subsequent model development and training.