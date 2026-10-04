#!/usr/bin/env python3
"""
inference2.py — Pretrained-only inference for PhysicsSLM.

Purpose
-------
Evaluate a PRETRAINED (not fine-tuned) PhysicsSLM checkpoint in isolation, so
that it can later be compared, apples-to-apples, against:

    1. Pretrained model (11,500 steps)
    2. Pretrained model (20,500 steps)
    3. Fine-tuned model (11,500 pretraining)
    4. Fine-tuned model (20,500 pretraining)

The architecture is ALWAYS taken from the checkpoint's own "config" field —
never hardcoded — and weights are always loaded with strict=True so that any
architecture mismatch fails loudly instead of silently.

This script intentionally contains no benchmarking code, no evaluation
metrics, and no fine-tuning logic. It only runs inference.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from model2 import PhysicsSLM

try:
    import sentencepiece as spm
except ImportError:
    spm = None


# ==========================================================================
# Prompt template — EXACTLY the format used during pretraining. Do not change.
# ==========================================================================
ANSWER_TAG = "<ANSWER>"


def build_prompt(question: str) -> str:
    """Builds the pretraining-format prompt for a raw question string."""
    return f"<QUESTION>\n\n{question.strip()}\n\n{ANSWER_TAG}\n\n"


# ==========================================================================
# Config / state containers
# ==========================================================================
@dataclass
class GenerationSettings:
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.7
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    repetition_penalty: float = 1.1

    def describe(self) -> str:
        if not self.do_sample:
            return (f"greedy decoding | max_new_tokens={self.max_new_tokens} | "
                    f"repetition_penalty={self.repetition_penalty}")
        return (f"sampling | temperature={self.temperature} | top_k={self.top_k} | "
                f"top_p={self.top_p} | max_new_tokens={self.max_new_tokens} | "
                f"repetition_penalty={self.repetition_penalty}")


@dataclass
class TimingResult:
    encode_s: float = 0.0
    generate_s: float = 0.0
    decode_s: float = 0.0

    @property
    def total_s(self) -> float:
        return self.encode_s + self.generate_s + self.decode_s


@dataclass
class TokenStats:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    context_length: int = 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.generated_tokens

    @property
    def context_usage_pct(self) -> float:
        return 100.0 * self.total_tokens / max(1, self.context_length)


@dataclass
class SessionStats:
    """Running totals shown by the /stats command."""
    questions_answered: int = 0
    total_prompt_tokens: int = 0
    total_generated_tokens: int = 0
    total_time_s: float = 0.0

    def update(self, tokens: TokenStats, timing: TimingResult) -> None:
        self.questions_answered += 1
        self.total_prompt_tokens += tokens.prompt_tokens
        self.total_generated_tokens += tokens.generated_tokens
        self.total_time_s += timing.total_s


# ==========================================================================
# Robustness checks
# ==========================================================================
def verify_paths(checkpoint_path: str, tokenizer_path: str) -> None:
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(tokenizer_path):
        print(f"ERROR: tokenizer not found: {tokenizer_path}", file=sys.stderr)
        sys.exit(1)


def load_checkpoint_dict(checkpoint_path: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in ckpt:
        print(f"ERROR: checkpoint '{checkpoint_path}' has no 'config' field. "
              f"Refusing to guess an architecture.", file=sys.stderr)
        sys.exit(1)
    if "model_state_dict" not in ckpt:
        print(f"ERROR: checkpoint '{checkpoint_path}' has no 'model_state_dict' field.",
              file=sys.stderr)
        sys.exit(1)
    return ckpt


def load_model_from_checkpoint(checkpoint_path: str, device: str) -> tuple[PhysicsSLM, dict]:
    """
    Loads architecture + weights strictly from the checkpoint. The config is
    never hardcoded here: it comes directly from checkpoint["config"], so the
    script automatically matches whatever architecture a given checkpoint was
    actually trained with.
    """
    ckpt = load_checkpoint_dict(checkpoint_path)
    config = ckpt["config"]

    model = PhysicsSLM(config)
    total_params = sum(p.numel() for p in model.parameters())

    # strict=True by design: any missing/unexpected key or shape mismatch
    # must fail loudly rather than silently producing a subtly wrong model.
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    model = model.to(device)
    model = model.float()  # fp32: correct precision for Pascal (P5000) GPUs
    model.eval()

    ckpt["_total_params"] = total_params
    return model, ckpt


def load_tokenizer(tokenizer_path: str):
    if spm is None:
        print("ERROR: sentencepiece is not installed. `pip install sentencepiece`.",
              file=sys.stderr)
        sys.exit(1)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    return tokenizer


# ==========================================================================
# Model info printing
# ==========================================================================
def get_config_attr(config, name: str, default=None):
    """Config may be an object (dataclass-like) or a plain dict; support both."""
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def print_model_info(experiment_name: str, checkpoint_path: str, ckpt: dict,
                      device: str) -> None:
    config = ckpt["config"]
    step = ckpt.get("step", "unknown")

    hidden_size = get_config_attr(config, "hidden_size", get_config_attr(config, "d_model", "?"))
    n_layers = get_config_attr(config, "n_layers", get_config_attr(config, "num_layers", "?"))
    n_heads = get_config_attr(config, "n_heads", get_config_attr(config, "num_heads", "?"))
    n_kv_heads = get_config_attr(config, "n_kv_heads", get_config_attr(config, "num_kv_heads", n_heads))
    vocab_size = get_config_attr(config, "vocab_size", "?")
    context_len = get_config_attr(
        config,
        "max_position_embeddings",
        get_config_attr(config, "max_seq_len", get_config_attr(config, "seq_len", "?")),
    )

    device_desc = device
    if torch.cuda.is_available() and device.startswith("cuda"):
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        device_desc = f"{device} ({props.name}, {props.total_memory / 1024**3:.1f} GB)"

    print("=" * 60)
    print("Physics Small Language Model")
    print("=" * 60)
    print(f"Experiment       : {experiment_name}")
    print(f"Checkpoint       : {checkpoint_path}")
    print(f"Checkpoint Step  : {step}")
    print(f"Hidden Size      : {hidden_size}")
    print(f"Layers           : {n_layers}")
    print(f"Heads            : {n_heads}")
    print(f"KV Heads         : {n_kv_heads}")
    print(f"Vocabulary Size  : {vocab_size}")
    print(f"Context Length   : {context_len}")
    print(f"Total Parameters : {ckpt['_total_params']:,}")
    print(f"Device           : {device_desc}")
    print("=" * 60)


# ==========================================================================
# Generation with repetition penalty (top_k / top_p / temperature / greedy)
# ==========================================================================
@torch.no_grad()
def apply_repetition_penalty(logits: torch.Tensor, generated_ids: list[int],
                              penalty: float) -> torch.Tensor:
    if penalty == 1.0 or not generated_ids:
        return logits
    unique_ids = set(generated_ids)
    for tok_id in unique_ids:
        score = logits[0, tok_id]
        logits[0, tok_id] = score / penalty if score > 0 else score * penalty
    return logits


@torch.no_grad()
def top_k_top_p_filter(logits: torch.Tensor, top_k: Optional[int],
                        top_p: Optional[float]) -> torch.Tensor:
    logits = logits.clone()
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_value = torch.topk(logits, top_k)[0][..., -1, None]
        logits[logits < kth_value] = float("-inf")

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cum_probs > top_p
        # Always keep at least the first (highest-probability) token.
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        indices_to_remove = sorted_mask.scatter(1, sorted_idx, sorted_mask)
        logits[indices_to_remove] = float("-inf")

    return logits


@torch.no_grad()
def generate_tokens(model: PhysicsSLM, input_ids: torch.Tensor,
                     settings: GenerationSettings, eos_token_id: Optional[int],
                     device: str, context_length: int = 2048) -> list[int]:
    """
    Deterministic-by-default token generation. Greedy decoding is used unless
    settings.do_sample is True, matching the previous inference script's
    defaults so results stay comparable across checkpoints.

    Context-length protection: generation stops early (instead of erroring
    out inside the model or silently producing garbage from truncated
    positional embeddings) once prompt_len + generated tokens would exceed
    the model's context_length.
    """
    generated: list[int] = []
    cur_ids = input_ids
    prompt_len = input_ids.shape[1]

    if prompt_len >= context_length:
        print(f"WARNING: prompt_tokens ({prompt_len}) already meets or exceeds "
              f"context_length ({context_length}). Skipping generation.",
              file=sys.stderr)
        return generated

    max_generatable = context_length - prompt_len
    effective_max_new = min(settings.max_new_tokens, max_generatable)
    if effective_max_new < settings.max_new_tokens:
        print(f"WARNING: requested max_new_tokens={settings.max_new_tokens} would "
              f"exceed context_length={context_length} given prompt_tokens={prompt_len}. "
              f"Capping generation at {effective_max_new} new tokens.", file=sys.stderr)

    for _ in range(effective_max_new):
        out = model(cur_ids)
        logits = out["logits"] if isinstance(out, dict) else out
        next_token_logits = logits[:, -1, :].clone()

        next_token_logits = apply_repetition_penalty(
            next_token_logits, generated, settings.repetition_penalty
        )

        if settings.do_sample:
            next_token_logits = next_token_logits / max(1e-6, settings.temperature)
            next_token_logits = top_k_top_p_filter(
                next_token_logits, settings.top_k, settings.top_p
            )
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        token_id = int(next_token.item())
        generated.append(token_id)

        if eos_token_id is not None and token_id == eos_token_id:
            break

        cur_ids = torch.cat([cur_ids, next_token], dim=1)

    return generated


def warmup_cuda(model: PhysicsSLM, device: str, tokenizer=None) -> None:
    """
    Runs a tiny throwaway forward pass so that CUDA context init, kernel
    autotuning/JIT, and memory-allocator warmup happen before the first
    *timed* generation, rather than inflating that measurement.
    """
    if not (torch.cuda.is_available() and str(device).startswith("cuda")):
        return
    try:
        with torch.no_grad():
            dummy_ids = torch.zeros((1, 4), dtype=torch.long, device=device)
            model(dummy_ids)
        torch.cuda.synchronize()
    except Exception as exc:  # pragma: no cover - warmup is best-effort
        print(f"NOTE: CUDA warmup skipped due to: {exc}", file=sys.stderr)


# ==========================================================================
# Output formatting — Final Answer / Step-by-Step Explanation / Theory
# ==========================================================================
def split_answer_sections(text: str) -> dict[str, str]:
    """
    Splits generated text into the three pretraining-format sections.
    Sections that are missing or empty are still returned (as empty strings)
    so the caller can print them anyway.
    """
    sections = {"Final Answer": "", "Step-by-Step Explanation": "", "Theory": ""}
    markers = [
        ("Final Answer:", "Final Answer"),
        ("Step-by-Step Explanation:", "Step-by-Step Explanation"),
        ("Theory:", "Theory"),
    ]

    positions = []
    for marker_text, key in markers:
        idx = text.find(marker_text)
        if idx != -1:
            positions.append((idx, marker_text, key))
    positions.sort()

    for i, (idx, marker_text, key) in enumerate(positions):
        start = idx + len(marker_text)
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        sections[key] = text[start:end].strip()

    if not positions:
        # No recognizable section markers at all — dump everything under
        # Final Answer rather than silently discarding it.
        sections["Final Answer"] = text.strip()

    return sections


# Markers that indicate the model has started a new, unrelated turn (e.g.
# hallucinating a fresh "<QUESTION>" block) after already answering. Anything
# from the first such marker onward is runaway output and gets trimmed.
RUNAWAY_MARKERS = ["<QUESTION>", ANSWER_TAG]


def trim_runaway_output(text: str) -> str:
    """
    Trims text after the model's answer has effectively ended but generation
    kept going to max_new_tokens (e.g. the model starts hallucinating a new
    question/answer pair, or repeats itself into a degenerate loop).
    """
    earliest = len(text)
    for marker in RUNAWAY_MARKERS:
        idx = text.find(marker, 1)  # skip a marker at position 0, if any
        if idx != -1:
            earliest = min(earliest, idx)
    trimmed = text[:earliest].rstrip()

    # Collapse pathological runs of the exact same line (a classic
    # degenerate-repetition failure mode) down to a single occurrence,
    # leaving a note about how many times it repeated.
    lines = trimmed.split("\n")
    collapsed: list[str] = []
    repeat_count = 1
    for i, line in enumerate(lines):
        if collapsed and line == collapsed[-1] and line.strip() != "":
            repeat_count += 1
            continue
        if repeat_count > 3:
            collapsed[-1] += f"  [repeated {repeat_count}x total, trimmed]"
        repeat_count = 1
        collapsed.append(line)
    if repeat_count > 3:
        collapsed[-1] += f"  [repeated {repeat_count}x total, trimmed]"

    return "\n".join(collapsed)


def print_answer(text: str) -> None:
    text = trim_runaway_output(text)
    sections = split_answer_sections(text)
    for title in ("Final Answer", "Step-by-Step Explanation", "Theory"):
        print(f"\n--- {title} ---")
        print(sections[title] if sections[title] else "(empty)")


# ==========================================================================
# Logging (JSONL)
# ==========================================================================
def log_interaction(log_file: str, question: str, answer: str, checkpoint_path: str,
                     step, settings: GenerationSettings, timing: TimingResult,
                     tokens: TokenStats) -> None:
    record = {
        "timestamp": time.time(),
        "question": question,
        "answer": answer,
        "checkpoint": checkpoint_path,
        "step": step,
        "generation_settings": {
            "do_sample": settings.do_sample,
            "temperature": settings.temperature,
            "top_k": settings.top_k,
            "top_p": settings.top_p,
            "repetition_penalty": settings.repetition_penalty,
            "max_new_tokens": settings.max_new_tokens,
        },
        "latency_seconds": {
            "encode": timing.encode_s,
            "generate": timing.generate_s,
            "decode": timing.decode_s,
            "total": timing.total_s,
        },
        "token_stats": {
            "prompt_tokens": tokens.prompt_tokens,
            "generated_tokens": tokens.generated_tokens,
            "total_tokens": tokens.total_tokens,
            "context_usage_pct": tokens.context_usage_pct,
        },
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ==========================================================================
# Single-question answering (shared by --question mode and the REPL)
# ==========================================================================
def answer_question(model: PhysicsSLM, tokenizer, question: str,
                     settings: GenerationSettings, device: str,
                     context_length: int) -> tuple[str, TimingResult, TokenStats]:
    timing = TimingResult()

    t0 = time.perf_counter()
    prompt = build_prompt(question)
    prompt_ids = tokenizer.encode(prompt, out_type=int)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    timing.encode_s = time.perf_counter() - t0

    eos_id = tokenizer.eos_id() if hasattr(tokenizer, "eos_id") and tokenizer.eos_id() >= 0 else None

    t0 = time.perf_counter()
    generated_ids = generate_tokens(model, input_ids, settings, eos_id, device,
                                     context_length=context_length)
    timing.generate_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    answer_text = tokenizer.decode(generated_ids)
    timing.decode_s = time.perf_counter() - t0

    tokens = TokenStats(
        prompt_tokens=len(prompt_ids),
        generated_tokens=len(generated_ids),
        context_length=context_length,
    )
    return answer_text, timing, tokens


def print_timing_and_tokens(timing: TimingResult, tokens: TokenStats) -> None:
    print("\n--- Timing ---")
    print(f"Encoding Time  : {timing.encode_s:.3f}s")
    print(f"Generation Time: {timing.generate_s:.3f}s")
    print(f"Decoding Time  : {timing.decode_s:.3f}s")
    print(f"Total Time     : {timing.total_s:.3f}s")

    print("\n--- Token Statistics ---")
    print(f"Prompt Tokens   : {tokens.prompt_tokens}")
    print(f"Generated Tokens: {tokens.generated_tokens}")
    print(f"Total Tokens    : {tokens.total_tokens}")
    print(f"Context Usage   : {tokens.context_usage_pct:.1f}%")


# ==========================================================================
# Interactive REPL
# ==========================================================================
HELP_TEXT = """
Available commands:
  /help    Show this help message
  /info    Show model / checkpoint information
  /stats   Show cumulative session statistics
  /clear   Clear the screen
  /exit    Exit the program

Anything else typed is treated as a physics question.
""".strip()


def run_repl(model: PhysicsSLM, tokenizer, settings: GenerationSettings,
             device: str, context_length: int, experiment_name: str,
             checkpoint_path: str, ckpt: dict, log_file: Optional[str]) -> None:
    session = SessionStats()
    print("\nType /help for commands.\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            print("Exiting.")
            break
        elif user_input == "/help":
            print(HELP_TEXT)
            continue
        elif user_input == "/info":
            print_model_info(experiment_name, checkpoint_path, ckpt, device)
            continue
        elif user_input == "/stats":
            print("\n--- Session Statistics ---")
            print(f"Questions Answered : {session.questions_answered}")
            print(f"Total Prompt Tokens: {session.total_prompt_tokens}")
            print(f"Total Gen. Tokens  : {session.total_generated_tokens}")
            print(f"Total Time         : {session.total_time_s:.3f}s")
            continue
        elif user_input == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            continue
        elif user_input.startswith("/"):
            print(f"Unknown command: {user_input}. Type /help for commands.")
            continue

        answer_text, timing, tokens = answer_question(
            model, tokenizer, user_input, settings, device, context_length
        )
        print_answer(answer_text)
        print_timing_and_tokens(timing, tokens)
        session.update(tokens, timing)

        if log_file:
            log_interaction(log_file, user_input, answer_text, checkpoint_path,
                             ckpt.get("step", "unknown"), settings, timing, tokens)


# ==========================================================================
# CLI
# ==========================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretrained-only inference for PhysicsSLM (no fine-tuning, "
                    "no benchmarking, no evaluation metrics)."
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pretrain_step20500.pt",
                         help="Pretrained checkpoint to evaluate.")
    parser.add_argument("--tokenizer_model", type=str, default="tok_analysis_v2/spm_11000.model",
                         help="SentencePiece tokenizer model path.")
    parser.add_argument("--experiment_name", type=str, default="pt20500_pretrained",
                         help="Label shown in the model info header and log records.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--question", type=str, default=None,
                         help="Answer a single question and exit (skips the REPL).")
    parser.add_argument("--log_file", type=str, default=None,
                         help="Optional JSONL file to append interaction logs to.")

    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true",
                         help="Use sampling instead of greedy decoding.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for reproducible sampling (only matters "
                              "when --do_sample is set; greedy decoding is already "
                              "deterministic). Seeds Python, PyTorch CPU, and CUDA RNGs.")
    parser.add_argument("--no_warmup", action="store_true",
                         help="Skip the CUDA warmup forward pass before timed generation.")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    verify_paths(args.checkpoint, args.tokenizer_model)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = args.device
    model, ckpt = load_model_from_checkpoint(args.checkpoint, device)
    tokenizer = load_tokenizer(args.tokenizer_model)

    print_model_info(args.experiment_name, args.checkpoint, ckpt, device)

    context_length = get_config_attr(
        ckpt["config"],
        "max_position_embeddings",
        get_config_attr(
            ckpt["config"],
            "max_seq_len",
            get_config_attr(ckpt["config"], "seq_len", 2048),
        ),
    )
    if not isinstance(context_length, int):
        context_length = 2048  # safe fallback for display/usage-percent only

    settings = GenerationSettings(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    print(f"\nGeneration settings: {settings.describe()}")
    if args.seed is not None:
        print(f"Seed               : {args.seed}")

    if not args.no_warmup:
        warmup_cuda(model, device)

    if args.question is not None:
        answer_text, timing, tokens = answer_question(
            model, tokenizer, args.question, settings, device, context_length
        )
        print(f"\nQuestion: {args.question}")
        print_answer(answer_text)
        print_timing_and_tokens(timing, tokens)

        if args.log_file:
            log_interaction(args.log_file, args.question, answer_text, args.checkpoint,
                             ckpt.get("step", "unknown"), settings, timing, tokens)
        return

    print("=" * 60)
    print(f"Experiment : {args.experiment_name}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Step       : {ckpt.get('step', 'unknown')}")
    print("Ready for inference...")
    print("=" * 60)

    run_repl(model, tokenizer, settings, device, context_length,
             args.experiment_name, args.checkpoint, ckpt, args.log_file)


if __name__ == "__main__":
    main()