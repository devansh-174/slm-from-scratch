"""
Supervised fine-tuning script for PhysicsSLM — Experiment tracking build (v2).

Base: train2.py (same logging / checkpoint / scheduler / resume style).
Fine-tunes a pretrained checkpoint on instruction-formatted JSONL data with
loss masked to only the answer/explanation/theory span (Llama/Qwen-style
instruction tuning).

Precision: FP32 by default (Pascal P5000 GPUs have poor FP16 throughput / no
native BF16). --fp16 / --bf16 are exposed for future GPUs but off by default.
GPUs: DataParallel if multiple GPUs are visible.

Batch sizing: micro batch is NOT hardcoded. We build the real DataLoader,
pull a genuine batch (through the real collate_fn, on the real dataset), and
run one real forward+backward with the real optimizer attached. If that OOMs,
we shrink the batch size (8 -> 6 -> 4 -> 2 -> 1) and rebuild the DataLoader.
This is deliberately NOT a synthetic dummy-tensor probe.

EXPERIMENT TRACKING (v2 — everything else unchanged from the base script):
  - --checkpoint is now REQUIRED (no default). Forcing an explicit value
    avoids silently fine-tuning a stale checkpoint after pretraining has
    moved on to a later step.
  - --experiment_name defaults to "pt20800_sft_v1" (short, sortable naming:
    pt11500_sft_v1, pt20800_sft_v1, pt30000_sft_v1, ...).
  - Every experiment gets ONE self-contained folder:
        finetune_checkpoints/<experiment_name>/
            pretrained_checkpoint.pt   <- exact copy of the source checkpoint
            best_model.pt
            latest_model.pt
            final_model.pt
            epoch_NNN.pt
            best_generation.txt / .json
            config.json
            metadata.json
            benchmark/                <- empty, for later benchmark_predictions.jsonl etc.
            evaluation/               <- empty, for later evaluation_summary.json / .md / csv
            outputs/                  <- periodic generation samples
            logs/                     <- training_log.csv
    Nothing for this run is written outside that folder.
  - The pretrained checkpoint's step and parameter count are printed and
    verified against the live model before training starts.
"""

import os
import json
import math
import time
import shutil
import argparse
import random
import csv
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model2 import PhysicsSLM, PhysicsSLMConfig

try:
    import sentencepiece as spm
except ImportError:
    spm = None


# --------------------------------------------------------------------------
# Prompt template / masking
# --------------------------------------------------------------------------
ANSWER_TAG = "<ANSWER>"


def format_sample(example: dict):
    """
    Builds the prompt and target text for one JSONL record.

    `answer` may be either:
      - a plain string (older dataset format), in which case the target is
        built from the top-level `answer` / `explanation` / `theory` fields
        exactly as before, or
      - a dict (current dataset format), e.g.:
            "answer": {
                "explanation": "...",
                "given": "...",
                "formula": "...",
                "substitution": "...",
                "final_answer": "..."
            }
        in which case the target is assembled from whichever of those
        sub-fields are present.
    """
    instruction = (
        example.get("instruction")
        or example.get("question")
        or ""
    )
    instruction = str(instruction).strip()

    answer = example.get("answer", "")

    if isinstance(answer, dict):
        parts = []

        if answer.get("final_answer"):
            parts.append(f"Final Answer:\n{answer['final_answer']}")

        if answer.get("explanation"):
            parts.append(f"Step-by-Step Explanation:\n{answer['explanation']}")

        if answer.get("given"):
            parts.append(f"Given:\n{answer['given']}")

        if answer.get("formula"):
            parts.append(f"Formula:\n{answer['formula']}")

        if answer.get("substitution"):
            parts.append(f"Substitution:\n{answer['substitution']}")

        target_part = "\n\n".join(parts)

    else:
        answer = str(answer).strip()
        explanation = str(example.get("explanation", "")).strip()
        theory = str(example.get("theory", "")).strip()
        target_part = (
            f"Final Answer:\n{answer}\n\n"
            f"Step-by-Step Explanation:\n{explanation}\n\n"
            f"Theory:\n{theory}"
        )

    prompt_part = f"<QUESTION>\n\n{instruction}\n\n{ANSWER_TAG}\n\n"
    return prompt_part, target_part


def truncate_middle(ids, max_len, head_ratio=0.5):
    """
    Truncates a token id list to max_len by keeping the beginning and end and
    dropping tokens from the middle, instead of blindly cutting the front.
    """
    if len(ids) <= max_len:
        return ids
    head_len = int(max_len * head_ratio)
    tail_len = max_len - head_len
    if tail_len <= 0:
        return ids[:max_len]
    return ids[:head_len] + ids[-tail_len:]


# --------------------------------------------------------------------------
# Dataset — JSONL instruction/answer pairs, loss-masked before <ANSWER> (UNCHANGED)
# --------------------------------------------------------------------------
class SFTDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, seq_len: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.examples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"WARNING: skipping malformed JSON at {jsonl_path}:{line_num} ({e})")
                    continue
                self.examples.append(obj)

        print(f"Loaded {len(self.examples):,} examples from {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        prompt_part, target_part = format_sample(example)

        prompt_ids = self.tokenizer.encode(prompt_part, out_type=int)
        target_ids = self.tokenizer.encode(target_part, out_type=int)

        eos_id = self.tokenizer.eos_id() if hasattr(self.tokenizer, "eos_id") else -1
        if eos_id is not None and eos_id >= 0:
            target_ids = target_ids + [eos_id]

        max_prompt_len = max(1, self.seq_len - len(target_ids))
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = truncate_middle(prompt_ids, max_prompt_len)

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids[:]

        if len(input_ids) > self.seq_len:
            input_ids = input_ids[:self.seq_len]
            labels = labels[:self.seq_len]

        return input_ids, labels, prompt_part


def collate_fn(batch, pad_id: int):
    """Pads a batch of (input_ids, labels, prompt_text) to the longest sequence."""
    max_len = max(len(x[0]) for x in batch)
    input_batch = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    label_batch = torch.full((len(batch), max_len), -100, dtype=torch.long)
    prompts = [b[2] for b in batch]

    for i, (ids, labels, _) in enumerate(batch):
        input_batch[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        label_batch[i, :len(labels)] = torch.tensor(labels, dtype=torch.long)

    return input_batch, label_batch, prompts


# --------------------------------------------------------------------------
# LR schedule: linear warmup -> cosine decay (UNCHANGED)
# --------------------------------------------------------------------------
def build_lr_lambda(total_steps: int, warmup_ratio: float = 0.04, min_lr_ratio: float = 0.1):
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
# Checkpoint helpers (UNCHANGED schema)
# --------------------------------------------------------------------------
def save_checkpoint(path, model, optimizer, scheduler, step, epoch, best_val_loss,
                     patience_counter, batch_in_epoch=0, data_generator=None, scaler=None):
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    ckpt = {
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "config": raw_model.config,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng_state": random.getstate(),
    }
    if data_generator is not None:
        ckpt["data_generator_state"] = data_generator.get_state()
    if scaler is not None:
        ckpt["grad_scaler_state"] = scaler.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, load_training_state=True,
                     data_generator=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.load_state_dict(ckpt["model_state_dict"])

    if not load_training_state:
        return 0, 0, float("inf"), 0, 0

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if ckpt.get("torch_rng_state") is not None:
        torch.set_rng_state(ckpt["torch_rng_state"].cpu()
                             if hasattr(ckpt["torch_rng_state"], "cpu") else ckpt["torch_rng_state"])
    if ckpt.get("cuda_rng_state_all") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])
    if ckpt.get("python_rng_state") is not None:
        random.setstate(ckpt["python_rng_state"])
    if data_generator is not None and ckpt.get("data_generator_state") is not None:
        data_generator.set_state(ckpt["data_generator_state"])
    if scaler is not None and ckpt.get("grad_scaler_state") is not None:
        scaler.load_state_dict(ckpt["grad_scaler_state"])

    return (
        ckpt.get("step", 0),
        ckpt.get("epoch", 0),
        ckpt.get("best_val_loss", float("inf")),
        ckpt.get("patience_counter", 0),
        ckpt.get("batch_in_epoch", 0),
    )


@torch.no_grad()
def evaluate(model, val_loader, device, autocast_ctx, max_batches=50):
    model.eval()
    losses = []
    for i, (x, y, _) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast_ctx():
            out = model(x, labels=y)
            loss = out["loss"]
        if isinstance(model, nn.DataParallel):
            loss = loss.mean()
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def gpu_memory_str():
    if not torch.cuda.is_available():
        return "n/a"
    parts = []
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        free = total - reserved
        parts.append(f"gpu{i}: alloc={alloc:.2f}GB reserved={reserved:.2f}GB free={free:.2f}GB")
    return " | ".join(parts)


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# --------------------------------------------------------------------------
# Real-batch micro-batch size selection (UNCHANGED — not a synthetic probe)
# --------------------------------------------------------------------------
def find_micro_batch_size(model, optimizer, train_ds, pad_id, device, num_workers,
                           autocast_ctx, scaler, candidates=(8, 6, 4, 2, 1)):
    print("=" * 60)
    print("Probing GPU memory with REAL batches to select micro batch size...")

    for bs in candidates:
        try:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            probe_loader = DataLoader(
                train_ds, batch_size=bs, shuffle=True, num_workers=0,
                collate_fn=lambda b: collate_fn(b, pad_id),
            )
            x, y, _ = next(iter(probe_loader))
            x, y = x.to(device), y.to(device)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                out = model(x, labels=y)
                loss = out["loss"]
                if isinstance(model, nn.DataParallel):
                    loss = loss.mean()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            optimizer.zero_grad(set_to_none=True)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"  micro_batch_size={bs}: OK  ({gpu_memory_str()})")
            del x, y, out, loss, probe_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return bs
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  micro_batch_size={bs}: OOM")
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise

    raise RuntimeError("Could not fit even micro_batch_size=1 on this GPU with a real batch.")


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--val_jsonl", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None,
                         help="path to a finetune checkpoint to resume (restores optimizer/scheduler/step)")
    # No default anymore: you must always name the exact pretrained checkpoint
    # you mean to fine-tune from (e.g. checkpoints/latest_model.pt right after
    # continuing pretraining, or checkpoints/pretrain_step20800.pt). This
    # avoids silently picking up a stale checkpoint.
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="REQUIRED. Exact pretrained base checkpoint to fine-tune from (weights only). "
                              "e.g. --checkpoint checkpoints/latest_model.pt or "
                              "--checkpoint checkpoints/pretrain_step20800.pt")
    parser.add_argument("--out_dir", type=str, default="finetune_checkpoints",
                         help="base directory; actual run output goes to <out_dir>/<experiment_name>/")
    parser.add_argument("--experiment_name", type=str, default="pt20800_sft_v1",
                         help="short, sortable experiment id, e.g. pt11500_sft_v1, pt20800_sft_v1, "
                              "pt30000_sft_v1. Everything for this run is written under "
                              "<out_dir>/<experiment_name>/ as a self-contained folder.")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokenizer_model", type=str, default="tok_analysis_v2/spm_11000.model")

    # Hyperparameters — keep these fixed across pt11500 / pt20800 / pt30000 / ...
    # experiments so pretraining amount is the only variable being compared.
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--warmup_ratio", type=float, default=0.04)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--validate_every", type=int, default=500)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--effective_batch", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    parser.add_argument("--fp16", action="store_true", help="use fp16 autocast (not recommended on Pascal)")
    parser.add_argument("--bf16", action="store_true", help="use bf16 autocast (requires Ampere+)")
    parser.add_argument("--sample_generation", action="store_true",
                         help="use sampling instead of greedy decoding for periodic eval generations")

    args = parser.parse_args()

    if args.fp16 and args.bf16:
        raise SystemExit("Pass only one of --fp16 / --bf16, not both.")

    SEED = args.seed
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.benchmark = True

    # ---------------- Self-contained experiment directory ----------------
    out_dir = os.path.join(args.out_dir, args.experiment_name)
    outputs_dir = os.path.join(out_dir, "outputs")
    logs_dir = os.path.join(out_dir, "logs")
    benchmark_dir = os.path.join(out_dir, "benchmark")
    evaluation_dir = os.path.join(out_dir, "evaluation")
    for d in (out_dir, outputs_dir, logs_dir, benchmark_dir, evaluation_dir):
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT HEADER")
    print(f"  Experiment Name    : {args.experiment_name}")
    print(f"  Fine-tuning Dataset: train={args.train_jsonl} | val={args.val_jsonl}")
    print(f"  Output Directory   : {out_dir}")
    print("=" * 60)

    # ---------------- Autocast context ----------------
    if args.fp16:
        amp_dtype = torch.float16
        print("Precision: fp16 autocast enabled (--fp16). NOTE: not recommended on Pascal GPUs "
              "(P5000) which lack fast fp16 tensor-core throughput; provided for future hardware.")
    elif args.bf16:
        amp_dtype = torch.bfloat16
        print("Precision: bf16 autocast enabled (--bf16). NOTE: requires Ampere or newer; Pascal "
              "GPUs (P5000) do not support bf16 natively.")
    else:
        amp_dtype = None
        print("Precision: fp32 (default, correct for Pascal P5000 GPUs).")

    def autocast_ctx():
        if amp_dtype is not None and torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
        return nullcontext()

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and torch.cuda.is_available())
    if args.fp16:
        print("GradScaler enabled for fp16 training.")

    # ---------------- Tokenizer ----------------
    if spm is None:
        raise SystemExit("sentencepiece is not installed. `pip install sentencepiece`.")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.tokenizer_model)

    for tok_file in [args.tokenizer_model, args.tokenizer_model.replace(".model", ".vocab")]:
        if os.path.exists(tok_file):
            shutil.copy(tok_file, os.path.join(out_dir, os.path.basename(tok_file)))
            print(f"Copied {tok_file} -> {out_dir}/")
        else:
            print(f"WARNING: tokenizer file not found at {tok_file}, skipping copy")

    device = args.device
    print("=" * 60)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  [{i}] {props.name} | {props.total_memory / 1024**3:.1f} GB | "
                  f"Compute Capability {props.major}.{props.minor}")
    print(f"Random seed: {SEED}")
    print("=" * 60)

    # ---------------- Data ----------------
    train_ds = SFTDataset(args.train_jsonl, tokenizer, args.seq_len)
    val_ds = SFTDataset(args.val_jsonl, tokenizer, args.seq_len)

    if len(val_ds) == 0:
        raise SystemExit("val_jsonl produced 0 usable examples; fix the val split before training.")

    pad_id = tokenizer.pad_id() if tokenizer.pad_id() >= 0 else 0

    # ---------------- Model ----------------
    config = PhysicsSLMConfig()  # locked architecture, DO NOT modify
    model = PhysicsSLM(config).to(device)
    model = model.float()

    total_params = sum(p.numel() for p in model.parameters())
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print("=" * 60)

    # ---------------- Load + verify + archive the pretrained checkpoint ----------------
    pretrained_step_info = None
    if args.resume is None:
        if not os.path.exists(args.checkpoint):
            raise SystemExit(f"Base checkpoint not found: {args.checkpoint}")

        _peek_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        pretrained_step_info = _peek_ckpt.get("step", "unknown")

        # Count UNIQUE tensors in the checkpoint by storage pointer, not a
        # raw sum over state_dict values. Models with tied weights (e.g.
        # tie_weights=True sharing tok_embeddings.weight / lm_head.weight)
        # store the same underlying tensor under two keys, so a raw sum
        # double-counts it and produces a false mismatch against
        # model.parameters() (which PyTorch already de-duplicates).
        _seen_ptrs = set()
        _ckpt_param_count = 0
        for tensor in _peek_ckpt["model_state_dict"].values():
            ptr = tensor.untyped_storage().data_ptr()
            if ptr in _seen_ptrs:
                continue
            _seen_ptrs.add(ptr)
            _ckpt_param_count += tensor.numel()

        print("=" * 60)
        print("Loading pretrained checkpoint:")
        print(f"  {args.checkpoint}")
        print(f"  Pretraining Step : {pretrained_step_info}")
        print(f"  Checkpoint unique parameters : {_ckpt_param_count:,}")
        print(f"  Model parameters             : {total_params:,}")
        print("=" * 60)

        if _ckpt_param_count != total_params:
            raise SystemExit(
                f"Checkpoint parameter count mismatch: '{args.checkpoint}' has "
                f"{_ckpt_param_count:,} unique parameters but the current PhysicsSLMConfig "
                f"produces {total_params:,} parameters. Refusing to load a mismatched "
                f"checkpoint — verify you're pointing at the correct pretrained checkpoint."
            )
        del _peek_ckpt

        # Archive an exact copy of the pretrained checkpoint inside the
        # experiment folder, so months later it's unambiguous which
        # pretrained weights produced this fine-tune's results.
        archived_ckpt_path = os.path.join(out_dir, "pretrained_checkpoint.pt")
        shutil.copy(args.checkpoint, archived_ckpt_path)
        print(f"Archived exact copy of pretrained checkpoint -> {archived_ckpt_path}")

        load_checkpoint(args.checkpoint, model, optimizer=None, scheduler=None, load_training_state=False)
        print(f"Loaded pretrained weights from {args.checkpoint}")

        # Stronger compatibility check than parameter counts: verify every
        # expected weight key/shape was actually consumed with none missing
        # or left over. load_checkpoint() above already used strict loading
        # internally (raw_model.load_state_dict(ckpt["model_state_dict"]),
        # which defaults to strict=True and would have raised on its own if
        # anything were incompatible) — this second explicit strict=False
        # call just surfaces the missing/unexpected key lists for the log.
        raw_model_for_check = model.module if isinstance(model, nn.DataParallel) else model
        _missing, _unexpected = raw_model_for_check.load_state_dict(
            torch.load(archived_ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"],
            strict=False,
        )
        print(f"Missing keys: {_missing}")
        print(f"Unexpected keys: {_unexpected}")
        if _missing or _unexpected:
            raise SystemExit(
                "Checkpoint is not compatible with the current architecture "
                "(missing/unexpected keys above)."
            )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)

    # ---------------- Optimizer (built BEFORE batch-size probing) ----------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.05,
    )

    data_generator = torch.Generator()
    data_generator.manual_seed(SEED)

    start_step, start_epoch, best_val_loss, patience_counter, resume_batch_in_epoch = 0, 0, float("inf"), 0, 0
    if args.resume:
        start_step, start_epoch, best_val_loss, patience_counter, resume_batch_in_epoch = load_checkpoint(
            args.resume, model, optimizer, scheduler=None, load_training_state=True,
            data_generator=data_generator, scaler=scaler,
        )
        print(f"Resumed weights/optimizer/RNG from {args.resume} at step {start_step}, epoch {start_epoch}, "
              f"batch_in_epoch={resume_batch_in_epoch}, best_val_loss={best_val_loss:.4f}, "
              f"patience_counter={patience_counter} (scheduler restore deferred until after it's built)")

    # ---------------- Real-batch micro batch size + grad accumulation ----------------
    EFFECTIVE_BATCH = args.effective_batch
    if device == "cuda":
        micro_batch_size = find_micro_batch_size(model, optimizer, train_ds, pad_id, device,
                                                  args.num_workers, autocast_ctx, scaler)
    else:
        micro_batch_size = 1
        print("Non-CUDA device: defaulting micro_batch_size=1")

    grad_accum = max(1, math.ceil(EFFECTIVE_BATCH / micro_batch_size))
    print(f"Chosen micro_batch_size={micro_batch_size}, grad_accumulation={grad_accum}, "
          f"effective_batch={micro_batch_size * grad_accum}")
    print("=" * 60)

    train_loader = DataLoader(
        train_ds, batch_size=micro_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
        persistent_workers=args.num_workers > 0, prefetch_factor=2 if args.num_workers > 0 else None,
        generator=data_generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=micro_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
        persistent_workers=args.num_workers > 0, prefetch_factor=2 if args.num_workers > 0 else None,
    )

    print(f"train batches/epoch: {len(train_loader):,}  |  val batches: {len(val_loader):,}")

    MAX_EPOCHS = args.epochs
    EARLY_STOP_PATIENCE = args.patience
    VALIDATE_EVERY = args.validate_every
    CHECKPOINT_EVERY = args.checkpoint_every
    GRAD_CLIP = 1.0

    steps_per_epoch = max(1, len(train_loader) // grad_accum)
    total_steps = steps_per_epoch * MAX_EPOCHS
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    print(f"steps/epoch: {steps_per_epoch}  |  total_steps ({MAX_EPOCHS} epochs): {total_steps:,}  |  "
          f"warmup_steps: {warmup_steps}  |  patience: {EARLY_STOP_PATIENCE}")
    print("=" * 60)

    lr_lambda = build_lr_lambda(total_steps, warmup_ratio=args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.resume:
        _resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "scheduler_state_dict" in _resume_ckpt:
            scheduler.load_state_dict(_resume_ckpt["scheduler_state_dict"])
            print(f"Restored scheduler state from {args.resume}")
        del _resume_ckpt

    # ---------------- Reproducibility metadata ----------------
    def get_git_commit_hash():
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def get_cuda_version():
        if torch.cuda.is_available():
            return torch.version.cuda
        return None

    import sys as _sys
    repro_metadata = {
        "git_commit_hash": get_git_commit_hash(),
        "python_version": _sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_version": get_cuda_version(),
        "gpu_names": [torch.cuda.get_device_properties(i).name
                      for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
    }

    # ---------------- config.json ----------------
    run_config = {
        "experiment_name": args.experiment_name,
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "checkpoint": args.checkpoint,
        "pretrained_step": pretrained_step_info,
        "out_dir": out_dir,
        "seq_len": args.seq_len,
        "seed": SEED,
        "micro_batch_size": micro_batch_size,
        "grad_accum_steps": grad_accum,
        "effective_batch": micro_batch_size * grad_accum,
        "optimizer": "AdamW",
        "lr": args.lr,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.05,
        "grad_clip": GRAD_CLIP,
        "scheduler": "linear_warmup_cosine_decay",
        "warmup_ratio": args.warmup_ratio,
        "max_epochs": MAX_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "validate_every": VALIDATE_EVERY,
        "checkpoint_every": CHECKPOINT_EVERY,
        "precision": "fp16" if args.fp16 else ("bf16" if args.bf16 else "fp32"),
        "max_new_tokens": args.max_new_tokens,
        "sample_generation": args.sample_generation,
        "model_config": config.__dict__,
        "reproducibility": repro_metadata,
    }
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, default=str)
    print(f"Wrote {config_path}")

    # ---------------- metadata.json ----------------
    metadata = {
        "experiment_name": args.experiment_name,
        "pretrained_checkpoint_path": args.checkpoint if args.resume is None else args.resume,
        "pretrained_checkpoint_archived_copy": "pretrained_checkpoint.pt" if args.resume is None else None,
        "pretrained_step": pretrained_step_info,
        "training_start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        "git_commit_hash": repro_metadata["git_commit_hash"],
        "tokenizer_path": args.tokenizer_model,
        "model_parameter_count": total_params,
        "random_seed": SEED,
        "hyperparameters": {
            "lr": args.lr,
            "epochs": args.epochs,
            "warmup_ratio": args.warmup_ratio,
            "patience": args.patience,
            "validate_every": args.validate_every,
            "checkpoint_every": args.checkpoint_every,
            "effective_batch": args.effective_batch,
            "seq_len": args.seq_len,
            "micro_batch_size": micro_batch_size,
            "grad_accum_steps": grad_accum,
            "precision": "fp16" if args.fp16 else ("bf16" if args.bf16 else "fp32"),
        },
    }
    metadata_path = os.path.join(out_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Wrote {metadata_path}")

    # ---------------- CSV log (now inside the experiment folder) ----------------
    csv_path = os.path.join(logs_dir, "training_log.csv")
    csv_is_new = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if csv_is_new:
        csv_writer.writerow(["epoch", "step", "train_loss", "val_loss", "val_perplexity",
                              "learning_rate", "grad_norm", "gpu_memory",
                              "tokens_per_second", "elapsed", "eta"])
        csv_file.flush()

    # ---------------- Fixed evaluation prompts (UNCHANGED) ----------------
    EVAL_PROMPTS = [
        "Explain Newton's First Law with a real-life example.",
        "A 5 kg block is pushed by a force of 20 N.\nFind acceleration.",
        "Derive the three equations of motion.",
        "Explain Work-Energy Theorem with derivation.",
        "Explain Ohm's Law.",
        "Explain Electromagnetic Induction.",
        "Derive Lens Formula.",
        "Why is the sky blue?",
        "Explain Photoelectric Effect.",
        "Solve a Class 12 CBSE long-answer physics question using Final Answer, "
        "Stepwise Explanation, and Theory.",
    ]

    @torch.no_grad()
    def run_generation_and_save(step_num: int, epoch_num: int, is_best: bool):
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        raw_model.eval()
        txt_path = os.path.join(outputs_dir, f"step_{step_num:05d}.txt")
        json_path = os.path.join(outputs_dir, f"step_{step_num:05d}_validation.json")
        records = []

        with open(txt_path, "w", encoding="utf-8") as f:
            for prompt in EVAL_PROMPTS:
                prompt_text = f"<QUESTION>\n\n{prompt}\n\n{ANSWER_TAG}\n\n"
                ids = tokenizer.encode(prompt_text, out_type=int)
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                try:
                    if args.sample_generation:
                        gen = raw_model.generate(
                            input_ids,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=True,
                            temperature=0.7,
                            top_k=50,
                            top_p=0.9,
                            eos_token_id=tokenizer.eos_id() if tokenizer.eos_id() >= 0 else None,
                        )
                    else:
                        gen = raw_model.generate(
                            input_ids,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            temperature=1.0,
                            top_k=None,
                            top_p=None,
                            eos_token_id=tokenizer.eos_id() if tokenizer.eos_id() >= 0 else None,
                        )
                    new_tokens = gen[0][input_ids.shape[1]:]
                    text = tokenizer.decode(new_tokens.tolist())
                except Exception as e:
                    text = f"[generation failed: {e}]"

                f.write("=" * 60 + "\n")
                f.write(f"PROMPT: {prompt}\n")
                f.write("-" * 60 + "\n")
                f.write(text + "\n\n")

                records.append({
                    "step": step_num,
                    "epoch": epoch_num,
                    "prompt": prompt,
                    "prediction": text,
                    "timestamp": time.time(),
                })

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        print(f"Wrote generations to {txt_path} and {json_path}")

        if is_best:
            best_txt_path = os.path.join(out_dir, "best_generation.txt")
            best_json_path = os.path.join(out_dir, "best_generation.json")
            shutil.copy(txt_path, best_txt_path)
            shutil.copy(json_path, best_json_path)
            print(f"Updated {best_txt_path} and {best_json_path} (new best val loss)")

        raw_model.train()

    global_step = start_step
    model.train()
    tokens_seen_window = 0
    window_start_time = time.time()
    last_grad_norm = 0.0
    epoch = start_epoch
    i = 0

    def handle_interrupt_and_exit(epoch_num, batch_in_epoch):
        print("\nKeyboardInterrupt caught — saving interrupted_model.pt ...")
        save_checkpoint(os.path.join(out_dir, "interrupted_model.pt"), model, optimizer, scheduler,
                         global_step, epoch_num, best_val_loss, patience_counter, batch_in_epoch=batch_in_epoch, data_generator=data_generator, scaler=scaler)
        if not csv_file.closed:
            csv_file.close()
        elapsed = time.time() - start_time
        print(f"Saved. Elapsed: {elapsed/3600:.2f}h. Exiting.")

    try:
        for epoch in range(start_epoch, MAX_EPOCHS):
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for i, (x, y, _) in enumerate(train_loader):
                if epoch == start_epoch and i < resume_batch_in_epoch:
                    continue

                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                with autocast_ctx():
                    out = model(x, labels=y)
                    loss = out["loss"]
                    if isinstance(model, nn.DataParallel):
                        loss = loss.mean()

                if not torch.isfinite(loss):
                    print(f"WARNING: non-finite loss ({loss.item()}) at epoch {epoch} batch {i}, "
                          f"step {global_step} — skipping this batch.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss = loss / grad_accum
                scaler.scale(loss).backward()
                accum_loss += loss.item()

                tokens_seen_window += x.numel()

                if (i + 1) % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    last_grad_norm = float(grad_norm_tensor)

                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1
                    train_loss = accum_loss
                    accum_loss = 0.0

                    now = time.time()
                    window_elapsed = now - window_start_time
                    tok_per_sec = tokens_seen_window / max(1e-6, window_elapsed)
                    elapsed_total = now - start_time
                    frac_done = global_step / max(1, total_steps)
                    eta_seconds = (elapsed_total / frac_done - elapsed_total) if frac_done > 0 else 0

                    if global_step % 20 == 0:
                        print(
                            f"[{args.experiment_name}] [epoch {epoch}] [step {global_step}/{total_steps}] "
                            f"train_loss={train_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                            f"grad_norm={last_grad_norm:.4f} "
                            f"patience={patience_counter}/{EARLY_STOP_PATIENCE} "
                            f"tok/s={tok_per_sec:.0f} elapsed={format_eta(elapsed_total)} "
                            f"eta={format_eta(eta_seconds)} "
                            f"micro_bs={micro_batch_size} grad_accum={grad_accum} "
                            f"eff_bs={micro_batch_size * grad_accum}"
                        )
                        tokens_seen_window = 0
                        window_start_time = now

                    if global_step % VALIDATE_EVERY == 0:
                        val_loss = evaluate(model, val_loader, device, autocast_ctx, max_batches=args.eval_batches)
                        try:
                            val_ppl = math.exp(val_loss)
                        except OverflowError:
                            val_ppl = float("inf")
                        mem_str = gpu_memory_str()

                        is_best = val_loss < best_val_loss
                        if is_best:
                            best_val_loss = val_loss
                            patience_counter = 0
                            save_checkpoint(os.path.join(out_dir, "best_model.pt"), model, optimizer,
                                             scheduler, global_step, epoch, best_val_loss,
                                             patience_counter, batch_in_epoch=i + 1, data_generator=data_generator, scaler=scaler)
                        else:
                            patience_counter += 1

                        print(
                            f"[{args.experiment_name}] [step {global_step}] val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} "
                            f"best={best_val_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                            f"grad_norm={last_grad_norm:.4f} "
                            f"patience={patience_counter}/{EARLY_STOP_PATIENCE}"
                        )
                        print(f"GPU memory: {mem_str}")

                        csv_writer.writerow([epoch, global_step, f"{train_loss:.6f}", f"{val_loss:.6f}",
                                              f"{val_ppl:.4f}", f"{scheduler.get_last_lr()[0]:.8e}",
                                              f"{last_grad_norm:.6f}", mem_str,
                                              f"{tok_per_sec:.1f}", f"{elapsed_total:.1f}", f"{eta_seconds:.1f}"])
                        csv_file.flush()

                        run_generation_and_save(global_step, epoch, is_best)

                        if patience_counter >= EARLY_STOP_PATIENCE:
                            print(f"Early stopping: no improvement for {EARLY_STOP_PATIENCE} validation checks.")
                            save_checkpoint(os.path.join(out_dir, "final_model.pt"), model, optimizer,
                                             scheduler, global_step, epoch, best_val_loss,
                                             patience_counter, batch_in_epoch=i + 1, data_generator=data_generator, scaler=scaler)
                            csv_file.close()
                            elapsed = time.time() - start_time
                            print("=" * 60)
                            print(f"[{args.experiment_name}] Training finished in {elapsed/3600:.2f} hours")
                            print("=" * 60)
                            return

                    if global_step % CHECKPOINT_EVERY == 0:
                        save_checkpoint(os.path.join(out_dir, "latest_model.pt"), model, optimizer,
                                         scheduler, global_step, epoch, best_val_loss,
                                         patience_counter, batch_in_epoch=i + 1, data_generator=data_generator, scaler=scaler)

            print(f"[{args.experiment_name}] Epoch {epoch} complete. global_step={global_step}")
            save_checkpoint(os.path.join(out_dir, f"epoch_{epoch + 1:03d}.pt"), model, optimizer,
                             scheduler, global_step, epoch + 1, best_val_loss,
                             patience_counter, batch_in_epoch=0, data_generator=data_generator, scaler=scaler)
            save_checkpoint(os.path.join(out_dir, "latest_model.pt"), model, optimizer,
                             scheduler, global_step, epoch + 1, best_val_loss,
                             patience_counter, batch_in_epoch=0, data_generator=data_generator, scaler=scaler)

        save_checkpoint(os.path.join(out_dir, "final_model.pt"), model, optimizer, scheduler,
                         global_step, MAX_EPOCHS, best_val_loss, patience_counter, batch_in_epoch=0, data_generator=data_generator, scaler=scaler)

        print(f"[{args.experiment_name}] Fine-tuning complete.")
        elapsed = time.time() - start_time
        print("=" * 60)
        print(f"[{args.experiment_name}] Training finished in {elapsed/3600:.2f} hours")
        print("=" * 60)

    except KeyboardInterrupt:
        handle_interrupt_and_exit(epoch, i + 1)
    finally:
        if not csv_file.closed:
            csv_file.close()


if __name__ == "__main__":
    main()