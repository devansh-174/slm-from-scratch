# Model Pretraining

## Overview

This stage pretrains the Physics Small Language Model using the binary datasets generated in the previous stage.

The model is trained using autoregressive next-token prediction with periodic validation, checkpointing, and learning rate scheduling.

---

## Features

- Binary dataset loading
- Autoregressive language modeling
- AdamW optimization
- Linear warmup + cosine decay
- Gradient accumulation
- Gradient clipping
- Validation & checkpointing
- Resume training
- Early stopping

---

## Workflow

```text
train.bin / val.bin
        │
        ▼
Dataset Loader
        │
        ▼
PhysicsSLM
        │
        ▼
Training
        │
        ▼
Validation
        │
        ▼
Checkpoint
```

---

## Outputs

- `best.pt`
- `latest.pt`
- Training checkpoints
- Training logs

---

## Outcome

Produces the pretrained PhysicsSLM checkpoint for inference and downstream fine-tuning.
