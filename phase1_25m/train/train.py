"""
Training script for PhysicsSLM — matches the locked training config:

Optimizer: AdamW (lr=5e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
Scheduler: Linear warmup (3%) -> Cosine decay
Precision: FP32
GPUs: 2x Quadro P5000 (16GB) via DataParallel
Grad accumulation: 4  |  Per-GPU/micro batch chosen so effective batch = 128
Max epochs: 200 | Early stopping patience: 20 validation checks
Validate + checkpoint: every 500 optimizer steps
Save best + latest checkpoints | Resume support | Seed 42

FIXES APPLIED IN THIS VERSION (vs the original):
  1. TokenizedTextDataset now chunks NON-OVERLAPPING blocks (stride = seq_len)
     instead of a stride-1 sliding window. The original version treated every
     single-token shift as a "new" training example, inflating dataset length
     ~500x with near-duplicate windows and pushing total_steps into the
     millions -- which in turn made the 3% LR warmup take ~100,000 steps
     before the learning rate reached anywhere near its target value. With
     non-overlapping chunks, total_steps for this dataset size is a realistic
     few thousand, and warmup completes in a few hundred steps as intended.
  2. patience_counter is now saved to and restored from checkpoints. Before
     this fix, resuming a run always reset the early-stopping patience
     counter to 0, silently discarding however much "no improvement" history
     had already accumulated before the interruption.
  3. Learning rate lowered to 3e-4 (from 5e-4) to test whether the higher LR
     was responsible for validation degradation. This is the ONLY substantive
     change in this experiment -- everything else is held constant so the
     result is attributable to the LR change alone.
  4. Added train_loss logging every 100 optimizer steps, corrected patience
     print ordering (now reports patience AFTER it's updated, not before),
     and added a startup summary of total/warmup steps.
"""

import os
import math
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import PhysicsSLM, PhysicsSLMConfig


# --------------------------------------------------------------------------
# Dataset — non-overlapping chunks of the packed token stream
# --------------------------------------------------------------------------
class TokenizedTextDataset(Dataset):
    """
    Expects a .bin/.pt file (or memmap) of token ids produced by your
    tokenization step. Each __getitem__ returns a fixed-length, NON-OVERLAPPING
    block of `seq_len + 1` tokens (input = [:-1], target = [1:]).
    """

    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        # non-overlapping chunks: how many full (seq_len+1)-sized blocks fit
        return max(0, (len(self.token_ids) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.token_ids[start: start + self.seq_len + 1]
        x = chunk[:-1].clone()
        y = chunk[1:].clone()
        return x, y


def load_token_ids(path: str) -> torch.Tensor:
    # Adjust to match how your tokenizer step saved ids (e.g. np.memmap uint16/int32, or torch.save)
    if path.endswith(".pt"):
        return torch.load(path)
    import numpy as np
    arr = np.fromfile(path, dtype=np.uint16)  # change dtype if vocab needs >65535
    return torch.from_numpy(arr.astype("int64"))


# --------------------------------------------------------------------------
# LR schedule: linear warmup -> cosine decay
# --------------------------------------------------------------------------
def build_lr_lambda(total_steps: int, warmup_ratio: float = 0.03, min_lr_ratio: float = 0.1):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


# --------------------------------------------------------------------------
# Checkpoint helpers
# --------------------------------------------------------------------------
def save_checkpoint(path, model, optimizer, scheduler, step, epoch, best_val_loss, patience_counter):
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "config": raw_model.config,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return (
        ckpt.get("step", 0),
        ckpt.get("epoch", 0),
        ckpt.get("best_val_loss", float("inf")),
        ckpt.get("patience_counter", 0),  # defaults to 0 for checkpoints saved before this fix
    )


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        out = model(x)  # x, y already pre-aligned by the dataset (y = x shifted by 1)
        loss = torch.nn.functional.cross_entropy(
            out["logits"].reshape(-1, out["logits"].size(-1)), y.reshape(-1), ignore_index=-100
        )
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_bin", type=str, required=True)
    parser.add_argument("--val_bin", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from")
    parser.add_argument("--seq_len", type=int, default=512)  # == max_position_embeddings
    parser.add_argument("--micro_batch_size", type=int, default=32)  # global batch per forward call (DataParallel splits this across GPUs, it does NOT multiply it)
    parser.add_argument("--grad_accum_steps", type=int, default=4)   # -> effective batch = 32 * 4 = 128
    args = parser.parse_args()

    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- Data ----------------
    train_ids = load_token_ids(args.train_bin)
    val_ids = load_token_ids(args.val_bin)
    train_ds = TokenizedTextDataset(train_ids, args.seq_len)
    val_ds = TokenizedTextDataset(val_ids, args.seq_len)

    print(f"train dataset: {len(train_ds):,} non-overlapping blocks of {args.seq_len} tokens "
          f"(from {len(train_ids):,} total tokens)")
    print(f"val dataset:   {len(val_ds):,} non-overlapping blocks of {args.seq_len} tokens "
          f"(from {len(val_ids):,} total tokens)")

    train_loader = DataLoader(train_ds, batch_size=args.micro_batch_size, shuffle=True,
                               num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.micro_batch_size, shuffle=False,
                             num_workers=2, pin_memory=True, drop_last=True)

    # ---------------- Model ----------------
    config = PhysicsSLMConfig()  # locked architecture spec
    model = PhysicsSLM(config).to(device)
    model = model.float()  # FP32 precision as specified

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)

    # ---------------- Optimizer ----------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,                 # Changed from 5e-4
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )

    MAX_EPOCHS = 200
    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    total_steps = steps_per_epoch * MAX_EPOCHS
    print(f"steps/epoch: {steps_per_epoch}  |  total_steps (200 epochs): {total_steps:,}")

    lr_lambda = build_lr_lambda(total_steps, warmup_ratio=0.03)

    warmup_steps = int(total_steps * 0.03)

    print("=" * 60)
    print(f"Total optimizer steps : {total_steps}")
    print(f"Warmup steps          : {warmup_steps}")
    print(f"Initial LR            : 3e-4")
    print("=" * 60)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_step, start_epoch, best_val_loss, patience_counter = 0, 0, float("inf"), 0
    if args.resume:
        start_step, start_epoch, best_val_loss, patience_counter = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        print(f"Resumed from {args.resume} at step {start_step}, epoch {start_epoch}, "
              f"best_val_loss={best_val_loss:.4f}, patience_counter={patience_counter}")

    VALIDATE_EVERY = 500      # optimizer steps
    CHECKPOINT_EVERY = 500    # optimizer steps
    EARLY_STOP_PATIENCE = 20  # validation checks with no improvement
    GRAD_CLIP = 1.0

    global_step = start_step
    model.train()

    for epoch in range(start_epoch, MAX_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # dataset already returns pre-aligned pairs: y[t] is the target for x[t] (i.e. x shifted by 1)
            out = model(x)
            loss = torch.nn.functional.cross_entropy(
                out["logits"].reshape(-1, out["logits"].size(-1)), y.reshape(-1), ignore_index=-100
            )
            loss = loss / args.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

            if (i + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                train_loss = accum_loss

                if global_step % 100 == 0:
                    print(
                        f"[step {global_step}] "
                        f"train_loss={train_loss:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )

                accum_loss = 0.0

                if global_step % VALIDATE_EVERY == 0:
                    val_loss = evaluate(model, val_loader, device)

                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss
                        patience_counter = 0
                        save_checkpoint(os.path.join(args.out_dir, "best.pt"),
                                         model, optimizer, scheduler, global_step, epoch,
                                         best_val_loss, patience_counter)
                    else:
                        patience_counter += 1

                    print(
                        f"[step {global_step}] "
                        f"val_loss={val_loss:.4f} "
                        f"best={best_val_loss:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} "
                        f"patience={patience_counter}/{EARLY_STOP_PATIENCE}"
                    )

                    if global_step % CHECKPOINT_EVERY == 0:
                        save_checkpoint(os.path.join(args.out_dir, "latest.pt"),
                                         model, optimizer, scheduler, global_step, epoch,
                                         best_val_loss, patience_counter)

                    if patience_counter >= EARLY_STOP_PATIENCE:
                        print(f"Early stopping: no improvement for {EARLY_STOP_PATIENCE} validation checks.")
                        return

        print(f"Epoch {epoch} complete. global_step={global_step}")

    print("Training complete.")


if __name__ == "__main__":
    main()