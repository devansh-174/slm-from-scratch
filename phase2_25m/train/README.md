# `train.py` — Phase 2 SLM Training

Training script for the **Phase 2 Physics Small Language Model (~25M parameters)**. It handles model training, validation, gradient accumulation, checkpointing, early stopping, and resume support.

## Configuration

| Parameter             |                        Value |
| --------------------- | ---------------------------: |
| Optimizer             |                        AdamW |
| Learning rate         |                       `5e-4` |
| Weight decay          |                       `0.01` |
| Scheduler             | Linear warmup → Cosine decay |
| Warmup                |                         `3%` |
| Precision             |                         FP32 |
| Context length        |                      `1,400` |
| Micro batch           |                          `8` |
| Gradient accumulation |                          `4` |
| Effective batch       |                         `32` |
| Gradient clipping     |                        `1.0` |
| Max epochs            |                        `100` |
| Validation interval   |                  `500` steps |
| Checkpoint interval   |                  `500` steps |
| Early stopping        |             `15` validations |
| GPUs                  |           `2 × Quadro P5000` |
| Seed                  |                         `42` |

## Input

The script requires tokenized training and validation data:

```text
train1.bin
validation1.bin
```

or equivalent paths supplied through:

```bash
--train_bin
--val_bin
```

Binary token IDs are loaded as `uint16` and converted to `int64`.

## Training

The dataset uses **non-overlapping blocks** of `seq_len` tokens to avoid excessive duplication from stride-1 sliding windows. Each sample contains `seq_len + 1` tokens and is split into input/target sequences for next-token prediction.

Default:

```text
seq_len            : 1400
micro_batch_size   : 8
grad_accum_steps   : 4
```

## Validation & Early Stopping

Validation runs every `500` optimizer steps, using up to `50` validation batches. Training stops after `15` consecutive validation checks without improvement.

## Checkpoints

The script produces:

```text
checkpoints/
├── best_model.pt
├── latest_model.pt
└── final_model.pt
```

* `best_model.pt` — best validation loss
* `latest_model.pt` — latest resumable checkpoint
* `final_model.pt` — final checkpoint after training

Checkpoints include model, optimizer, scheduler, training progress, validation loss, and early-stopping state.

## Resume

Resume an interrupted run with:

```bash
python3 train.py \
    --train_bin train.bin \
    --val_bin validation.bin \
    --resume checkpoints/latest_model.pt
```

The script restores model, optimizer, scheduler, epoch, global step, batch position, best validation loss, and patience state.

## Usage

```bash
python3 train.py \
    --train_bin train.bin \
    --val_bin validation.bin \
    --out_dir checkpoints \
    --seq_len 1400 \
    --micro_batch_size 8 \
    --grad_accum_steps 4
```

## Model

The model is defined in:

```text
model1.py
```

and instantiated through `PhysicsSLMConfig` / `PhysicsSLM`. The Phase-2 configuration targets approximately **25M parameters**, with an 11,000-token vocabulary and 1,400-token context.
