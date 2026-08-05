# Model Inference

## Overview

This stage performs inference using the pretrained Physics Small Language Model.

The script loads the trained checkpoint and tokenizer, accepts user questions, generates responses using autoregressive decoding, and supports both interactive and single-question inference.

---

## Features

- Checkpoint loading
- Tokenizer loading
- Interactive inference (REPL)
- Single-question inference
- Configurable decoding
- Automatic context handling
- Response logging
- Generation statistics

---

## Workflow

```text
User Question
      │
      ▼
Tokenization
      │
      ▼
Load Checkpoint
      │
      ▼
PhysicsSLM
      │
      ▼
Text Generation
      │
      ▼
Decoded Response
```

---

## Outputs

- Generated answers
- Interactive inference session
- Optional JSONL response logs

---

## Outcome

Provides an interface for evaluating the pretrained PhysicsSLM by generating responses to user queries and analyzing model behavior.
