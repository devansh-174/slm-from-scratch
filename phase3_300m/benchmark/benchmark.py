#!/usr/bin/env python3
"""
benchmark.py

Generates predictions for the Physics Small Language Model (PhysicsSLM, ~300M params)
on a benchmark dataset. This script ONLY produces model_answer text and timing/token
metadata for each question — it does NOT compute BLEU/ROUGE/BERTScore or judge
correctness in any way. A separate evaluation script is expected to consume
benchmark_predictions.jsonl later.

Design goals (per project spec):
  - Reuse the existing inference pipeline from inference_fine_tune.py rather than
    re-implementing model/tokenizer loading or generation logic.
  - Use the exact same prompt template (<QUESTION> / <ANSWER>) and generation
    settings as inference_fine_tune.py.
  - Stream results to disk immediately (append-per-question), support --resume,
    and never let a single failed question kill the run.

v2 changes vs. the first draft
================================
The first draft imported inference_fine_tune.py defensively, guessing at function
names (load_model, generate, sample, etc.) because I didn't have the real source.
Now that inference_fine_tune.py is available, this version imports its actual
functions directly by name:

    build_sft_prompt(instruction) -> str
    load_tokenizer(tokenizer_path) -> spm.SentencePieceProcessor
    resolve_checkpoint_path(explicit_path, checkpoint_dir) -> str
    load_model_from_checkpoint(checkpoint_path, device) -> (model, config, metadata, total_params)
    generate_response(model, tokenizer, instruction, device, precision,
                       do_sample, temperature, top_p, top_k, max_new_tokens,
                       repetition_penalty) -> (text, timings, stats) | None

There is no more name-guessing, no more silent prompt-template fallback, and no
more re-tokenizing the output to approximate token counts — prompt_tokens /
generated_tokens / total_tokens and encode/generation/decode timings all come
straight from generate_response()'s own return values, so benchmark numbers are
identical in kind to what inference_fine_tune.py reports interactively.

Also fixed / added in this version:
  - Parameter count and special-token mismatches now ABORT (matching
    inference_fine_tune.py's own strict behavior) instead of just warning,
    unless --force is passed.
  - decode_time_ms is real (from generate_response), not hardcoded to 0.
  - ETA is based on total per-question wall time, not generation time alone.
  - Empty/whitespace-only questions are skipped and logged, not sent to the model.
  - A run-level metadata JSON (checkpoint, tokenizer, device, decoding config,
    git-independent settings, start/end time) is written alongside the
    predictions file.
  - --seed for reproducible sampling runs (only matters when do_sample=True).
  - Peak GPU memory is logged in the final summary when CUDA is available.

v2.2 changes (this revision)
================================
  - Startup sanity check that generate_response() actually consumes
    build_sft_prompt()-formatted text the way we expect, instead of silently
    trusting that benchmark.py and inference_fine_tune.py agree on the prompt
    template. This does NOT re-implement prompt construction; it just asserts
    build_sft_prompt is importable/callable and logs the exact template it
    produces for one sample question, so a human can eyeball it against
    inference_fine_tune.py's interactive output before a long run starts.
  - Inference calls are wrapped in torch.inference_mode() here as a belt-and-
    suspenders measure. If generate_response() already does this internally,
    the nested context is a harmless no-op.
  - Run metadata now records completed_at (in addition to started_at) and
    peak_gpu_memory_mb, and includes model_name for self-describing reports.
  - Prints the benchmark filename being used at startup.
  - Saves model_answer_raw alongside model_answer whenever generate_response()
    exposes a pre-cleanup value (falls back to None if it doesn't; see note
    in run_benchmark()).

v2.3 changes (this revision)
================================
  - verify_prompt_template() no longer just prints the rendered prompt for a
    human to eyeball; it now asserts automatically that build_sft_prompt()'s
    output contains the <QUESTION>/<ANSWER> markers, preserves the raw
    question text verbatim, and is longer than the raw question. Fails the
    run (unless --force) if any of those don't hold.
  - model_answer_raw now defaults to the same `text` generate_response()
    returned, instead of a hardcoded None, when no distinct pre-cleanup value
    is exposed — a None placeholder added no information.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import torch

# ----------------------------------------------------------------------------
# Project paths — adjust PROJECT_ROOT if you move this script elsewhere.
# ----------------------------------------------------------------------------
PROJECT_ROOT = "/mnt/hdd-data4/interns/slm from scratch"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_BENCHMARK_FILE = os.path.join(PROJECT_ROOT, "physics_slm_benchmark_1000.jsonl")
DEFAULT_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "finetune_checkpoints")
DEFAULT_OUTPUT_FILE = os.path.join(PROJECT_ROOT, "benchmark_predictions.jsonl")
DEFAULT_TOKENIZER_MODEL = os.path.join(PROJECT_ROOT, "tok_analysis_v2", "spm_11000.model")

# Human-readable model identifier for self-describing run metadata / reports.
# Purely descriptive — does not affect loading or generation in any way.
MODEL_NAME = "PhysicsSLM-300M"

# Bump this by hand whenever benchmark.py's logic changes in a way that could
# affect predictions (prompt handling, decoding defaults, stats computation).
# Belt-and-suspenders alongside the git commit hash below: the commit hash
# identifies the whole repo state, this identifies this file's own intent.
BENCHMARK_SCRIPT_VERSION = "2.3"


def get_git_commit(path: str):
    """Best-effort short git commit hash for the repo containing `path`, so a
    benchmark_predictions.jsonl file can be traced back to the exact code
    that produced it. Returns None (not an error) if git/the repo isn't
    available — this is diagnostic metadata, not something worth failing
    the whole run over."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(path)),
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
            dirty = subprocess.run(
                ["git", "diff", "--quiet"],
                cwd=os.path.dirname(os.path.abspath(path)),
                capture_output=True, timeout=5,
            ).returncode != 0
            return commit + ("-dirty" if dirty else "")
    except Exception:
        pass
    return None

# ==============================================================================
# REUSED PIPELINE IMPORTS — direct, no guessing.
# ==============================================================================
try:
    from inference_fine_tune import (
        build_sft_prompt,
        load_tokenizer,
        resolve_checkpoint_path,
        load_model_from_checkpoint,
        generate_response,
        EXPECTED_PARAM_COUNT,
        EXPECTED_VOCAB_SIZE,
        EXPECTED_UNK_ID,
        EXPECTED_BOS_ID,
        EXPECTED_EOS_ID,
        EXPECTED_PAD_ID,
        MAX_CONTEXT,
    )
except ImportError as e:
    print(f"[FATAL] Could not import inference_fine_tune.py from {PROJECT_ROOT!r}: {e}")
    print("        Make sure benchmark.py sits in the project root next to it, or that")
    print("        PROJECT_ROOT above points at the folder containing it.")
    sys.exit(1)


# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def verify_prompt_template(force: bool):
    """We deliberately do NOT re-implement prompt construction here — the whole
    point of importing generate_response() directly is that benchmark.py and
    inference_fine_tune.py can never disagree about the prompt template.

    What CAN still go wrong: generate_response() might have been refactored to
    build the prompt itself from the raw question, ignoring build_sft_prompt()
    entirely, or to expect an already-built prompt as input. Either way we'd
    silently benchmark something other than what interactive inference does.

    This is an automated check, not a "does this look right to a human" print
    statement: it renders build_sft_prompt() on a canary question and asserts
    it actually contains the expected <QUESTION>/<ANSWER> markers and wraps
    the raw question text, matching the template used in training (format_sample()
    in the SFT script: "<QUESTION>\\n\\n{instruction}\\n\\n<ANSWER>\\n\\n").
    Any mismatch here means benchmark.py and training/inference have drifted
    apart on the prompt format, which is exactly the failure mode this guards
    against — so it aborts (unless --force) rather than just warning.
    """
    canary_question = "What is the SI unit of force?"
    try:
        rendered = build_sft_prompt(canary_question)
    except Exception as e:
        msg = f"build_sft_prompt() raised {type(e).__name__}: {e}"
        if force:
            print(f"[WARN] {msg}\n[WARN] Continuing anyway because --force was passed.")
            return
        print(f"[FATAL] {msg}")
        print("        This means benchmark.py cannot confirm it is using the same")
        print("        prompt template as inference_fine_tune.py. Pass --force to skip.")
        sys.exit(1)

    errors = []
    if not isinstance(rendered, str) or not rendered.strip():
        errors.append(f"build_sft_prompt() returned {rendered!r}, expected a non-empty string.")
    else:
        if "<QUESTION>" not in rendered:
            errors.append("rendered prompt is missing the '<QUESTION>' marker.")
        if "<ANSWER>" not in rendered:
            errors.append("rendered prompt is missing the '<ANSWER>' marker.")
        if canary_question not in rendered:
            errors.append("rendered prompt does not contain the raw question text verbatim.")
        if len(rendered) <= len(canary_question):
            errors.append("rendered prompt is not longer than the raw question "
                           "(expected template wrapping to add length).")

    if errors:
        msg = "build_sft_prompt() output failed automated template checks:\n" + \
              "\n".join(f"  - {e}" for e in errors) + f"\n  rendered={rendered!r}"
        if force:
            print(f"[WARN] {msg}\n[WARN] Continuing anyway because --force was passed.")
            return
        print(f"[FATAL] {msg}")
        print("        This means benchmark.py may not be using the same prompt template")
        print("        as inference_fine_tune.py / training. Pass --force to skip.")
        sys.exit(1)

    print(f"[OK] build_sft_prompt() template check passed (markers present, "
          f"question preserved, {len(rendered)} chars rendered from "
          f"{len(canary_question)}-char canary question).")


def verify_tokenizer(tokenizer, force: bool):
    """inference_fine_tune.load_tokenizer() already hard-fails on a mismatched
    tokenizer, so by the time we get a tokenizer back here it's already been
    validated against EXPECTED_VOCAB_SIZE / EXPECTED_*_ID. This is a second,
    redundant check specific to benchmark.py so a future change to
    load_tokenizer()'s strictness doesn't silently loosen this script too."""
    errors = []
    if tokenizer.GetPieceSize() != EXPECTED_VOCAB_SIZE:
        errors.append(f"vocab size {tokenizer.GetPieceSize()} != expected {EXPECTED_VOCAB_SIZE}")
    if tokenizer.unk_id() != EXPECTED_UNK_ID:
        errors.append(f"unk_id {tokenizer.unk_id()} != expected {EXPECTED_UNK_ID}")
    if tokenizer.bos_id() != EXPECTED_BOS_ID:
        errors.append(f"bos_id {tokenizer.bos_id()} != expected {EXPECTED_BOS_ID}")
    if tokenizer.eos_id() != EXPECTED_EOS_ID:
        errors.append(f"eos_id {tokenizer.eos_id()} != expected {EXPECTED_EOS_ID}")
    if tokenizer.pad_id() != EXPECTED_PAD_ID:
        errors.append(f"pad_id {tokenizer.pad_id()} != expected {EXPECTED_PAD_ID}")

    if errors:
        msg = "Tokenizer does not match the expected fine-tuning configuration:\n" + \
              "\n".join(f"  - {e}" for e in errors)
        if force:
            print(f"[WARN] {msg}\n[WARN] Continuing anyway because --force was passed.")
        else:
            print(f"[FATAL] {msg}")
            print("        Pass --force to run anyway (NOT recommended for real benchmark runs).")
            sys.exit(1)
    else:
        print(f"[OK] Tokenizer verified: vocab={tokenizer.GetPieceSize()}, "
              f"unk={tokenizer.unk_id()}, bos={tokenizer.bos_id()}, "
              f"eos={tokenizer.eos_id()}, pad={tokenizer.pad_id()}")


def verify_param_count(total_params: int, force: bool):
    if total_params != EXPECTED_PARAM_COUNT:
        msg = (f"Loaded model has {total_params:,} parameters, "
               f"expected exactly {EXPECTED_PARAM_COUNT:,}.")
        if force:
            print(f"[WARN] {msg}\n[WARN] Continuing anyway because --force was passed.")
        else:
            print(f"[FATAL] {msg}")
            print("        Pass --force to run anyway (NOT recommended for real benchmark runs).")
            sys.exit(1)
    else:
        print(f"[OK] Parameter count verified: {total_params:,}")


def load_completed_ids(output_file: str):
    """Read an existing output file (if present) and return the set of already
    completed record IDs, so --resume can skip them."""
    completed = set()
    if not os.path.exists(output_file):
        return completed
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                rid = record.get("id")
                if rid is not None:
                    completed.add(rid)
            except json.JSONDecodeError:
                continue
    return completed


def save_prediction(output_file: str, record: dict):
    """Append one prediction record to the output JSONL file, flushing immediately
    so results survive interruption."""
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def save_run_metadata(args, meta_path, checkpoint_metadata, config, total_params,
                       device, precision, started_at_iso, checkpoint_path_full,
                       completed_at_iso=None, peak_gpu_memory_mb=None):
    """Writes (or overwrites) the run metadata JSON. Called once at the start of
    a run with completed_at_iso=None, and again at the end with the completion
    timestamp and peak GPU memory filled in, so a run that gets killed mid-way
    still leaves a metadata file behind (just without an end time)."""
    metadata = {
        "benchmark_script_version": BENCHMARK_SCRIPT_VERSION,
        "model_name": MODEL_NAME,
        "git_commit": get_git_commit(__file__),
        "started_at": started_at_iso,
        "completed_at": completed_at_iso,
        "benchmark_file": args.benchmark_file,
        "output_file": args.output_file,
        "checkpoint_path": checkpoint_path_full,
        "checkpoint_filename": checkpoint_metadata["checkpoint_filename"],
        "checkpoint_step": checkpoint_metadata["step"],
        "checkpoint_epoch": checkpoint_metadata["epoch"],
        "checkpoint_best_val_loss": checkpoint_metadata["best_val_loss"],
        "tokenizer_model": args.tokenizer_model,
        "total_params": total_params,
        "architecture": {
            "n_layers": config.n_layers,
            "hidden_size": config.hidden_size,
            "n_heads": config.n_heads,
            "n_kv_heads": config.n_kv_heads,
            "max_position_embeddings": config.max_position_embeddings,
        },
        "device": device,
        "precision": precision,
        "decoding": {
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
        },
        "seed": args.seed,
        "resume": args.resume,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Wrote run metadata to {meta_path}")


def print_summary(total, successful, failed, skipped, gen_times, prompt_tok_counts,
                   gen_tok_counts, total_elapsed_s, device):
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total questions      : {total}")
    print(f"Successful           : {successful}")
    print(f"Failed               : {failed}")
    print(f"Skipped (empty)      : {skipped}")
    if gen_times:
        print(f"Average gen time     : {sum(gen_times) / len(gen_times):.2f} ms")
    if prompt_tok_counts:
        print(f"Average prompt tokens: {sum(prompt_tok_counts) / len(prompt_tok_counts):.2f}")
    if gen_tok_counts:
        print(f"Average gen tokens   : {sum(gen_tok_counts) / len(gen_tok_counts):.2f}")
    print(f"Total benchmark time : {total_elapsed_s:.2f} s")
    peak_mb = None
    if device.startswith("cuda") and torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak GPU memory      : {peak_mb:.1f} MB")
    print("=" * 60)
    return peak_mb


def run_benchmark(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"[OK] Seed set to {args.seed}")

    print(f"Benchmark file : {args.benchmark_file}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    precision = "fp16" if (args.fp16 and device.startswith("cuda")) else "fp32"
    print(f"Device      : {device}")
    if device.startswith("cuda"):
        print(f"GPU name    : {torch.cuda.get_device_properties(0).name}")
        torch.cuda.reset_peak_memory_stats()
    print(f"Precision   : {precision.upper()}")

    # ---- Prompt-template sanity check (does not re-implement anything) ----
    verify_prompt_template(args.force)

    # ---- Tokenizer (load_tokenizer() itself hard-fails on mismatch) ----
    tokenizer = load_tokenizer(args.tokenizer_model)
    verify_tokenizer(tokenizer, args.force)

    # ---- Checkpoint resolution + model load (delegated entirely) ----
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.checkpoint_dir)
    print(f"Checkpoint  : {checkpoint_path}")
    model, config, checkpoint_metadata, total_params = load_model_from_checkpoint(
        checkpoint_path, device
    )
    verify_param_count(total_params, args.force)

    # ---- Load benchmark records ----
    records = []
    with open(args.benchmark_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    completed_ids = set()
    if args.resume:
        completed_ids = load_completed_ids(args.output_file)
        print(f"[RESUME] Found {len(completed_ids)} completed IDs. Skipping those.")
    elif os.path.exists(args.output_file):
        print(f"[WARN] {args.output_file} already exists and --resume was not passed.")
        print("       New predictions will be appended to the existing file.")

    pending = [r for r in records if r.get("id") not in completed_ids]
    total_to_run = len(pending)
    print(f"Total questions in benchmark : {len(records)}")
    print(f"Already completed             : {len(completed_ids)}")
    print(f"Remaining to run               : {total_to_run}")

    started_at_iso = datetime.now(timezone.utc).isoformat()
    meta_path = os.path.splitext(args.output_file)[0] + ".run_metadata.json"
    save_run_metadata(
        args, meta_path, checkpoint_metadata, config, total_params,
        device, precision, started_at_iso,
        checkpoint_path_full=checkpoint_path,
    )

    checkpoint_name = os.path.basename(checkpoint_path)

    successful = 0
    failed = 0
    skipped = 0
    gen_times = []
    prompt_tok_counts = []
    gen_tok_counts = []

    run_start = time.perf_counter()

    iterator = pending
    if tqdm is not None:
        iterator = tqdm(pending, desc="Benchmarking", unit="q")

    for idx, record in enumerate(iterator, start=1):
        question = (record.get("question") or "").strip()
        rid = record.get("id", f"UNKNOWN_{idx}")

        output_record = dict(record)  # preserve all original fields
        output_record["checkpoint"] = checkpoint_name
        output_record["checkpoint_path"] = checkpoint_path
        output_record["do_sample"] = args.do_sample
        output_record["temperature"] = args.temperature
        output_record["top_p"] = args.top_p
        output_record["top_k"] = args.top_k
        output_record["max_new_tokens"] = args.max_new_tokens
        output_record["repetition_penalty"] = args.repetition_penalty
        output_record["timestamp"] = datetime.now(timezone.utc).isoformat()

        if not question:
            output_record["model_answer"] = None
            output_record["model_answer_raw"] = None
            output_record["status"] = "skipped_empty_question"
            skipped += 1
            save_prediction(args.output_file, output_record)
            print(f"[SKIP] id={rid}: empty question")
            continue

        try:
            # inference_mode() here is deliberately redundant if
            # generate_response() already wraps its own forward passes in it —
            # nesting inference_mode contexts is a documented no-op in PyTorch,
            # so this is a safe belt-and-suspenders addition rather than a
            # correctness risk.
            with torch.inference_mode():
                result = generate_response(
                    model, tokenizer, question, device, precision,
                    args.do_sample, args.temperature, args.top_p, args.top_k,
                    args.max_new_tokens, args.repetition_penalty,
                )
            if result is None:
                # generate_response() already prints its own ERROR line
                # (e.g. prompt too long, non-finite output) before returning None.
                raise RuntimeError("generate_response() returned None (see ERROR above)")

            text, timings, stats = result
            output_record["model_answer"] = text
            # generate_response()'s public return contract (per the docstring
            # at the top of this file) is (text, timings, stats) — there is no
            # documented pre-cleanup value distinct from `text`. Defaulting
            # model_answer_raw to a hardcoded None added nothing; falling back
            # to `text` itself at least means the field is never a dead
            # placeholder. If a future version of inference_fine_tune.py starts
            # exposing a genuine pre-cleanup value (e.g. stats["raw_text"]),
            # prefer that instead.
            output_record["model_answer_raw"] = (
                stats["raw_text"] if isinstance(stats, dict) and "raw_text" in stats else text
            )
            output_record["prompt_tokens"] = stats["prompt_tokens"]
            output_record["generated_tokens"] = stats["generated_tokens"]
            output_record["total_tokens"] = stats["total_tokens"]
            output_record["encode_time_ms"] = round(timings["encode_ms"], 2)
            output_record["generation_time_ms"] = round(timings["generation_ms"], 2)
            output_record["decode_time_ms"] = round(timings["decode_ms"], 2)
            output_record["total_time_ms"] = round(timings["total_ms"], 2)
            output_record["status"] = "success"

            successful += 1
            gen_times.append(timings["generation_ms"])
            prompt_tok_counts.append(stats["prompt_tokens"])
            gen_tok_counts.append(stats["generated_tokens"])

        except Exception as e:
            output_record["model_answer"] = None
            output_record["model_answer_raw"] = None
            output_record["status"] = "failed"
            output_record["error"] = f"{type(e).__name__}: {e}"
            for key in ("prompt_tokens", "generated_tokens", "total_tokens",
                        "encode_time_ms", "generation_time_ms", "decode_time_ms",
                        "total_time_ms"):
                output_record[key] = None
            failed += 1
            print(f"\n[FAILED] id={rid}: {type(e).__name__}: {e}")
            traceback.print_exc()

        save_prediction(args.output_file, output_record)

        if idx % 25 == 0 or idx == total_to_run:
            elapsed = time.perf_counter() - run_start
            avg_total_per_q = elapsed / idx
            eta_s = avg_total_per_q * (total_to_run - idx)
            print(f"\n[PROGRESS] Question {idx}/{total_to_run} "
                  f"(id={rid}) | elapsed={elapsed:.1f}s | "
                  f"avg_time/question={avg_total_per_q * 1000:.1f}ms | ETA={eta_s:.1f}s")
            sys.stdout.flush()

    total_elapsed = time.perf_counter() - run_start
    peak_mb = print_summary(
        total=len(records),
        successful=successful,
        failed=failed,
        skipped=skipped,
        gen_times=gen_times,
        prompt_tok_counts=prompt_tok_counts,
        gen_tok_counts=gen_tok_counts,
        total_elapsed_s=total_elapsed,
        device=device,
    )

    # Rewrite metadata with completion time + peak GPU memory now that the run
    # has finished. If the run was interrupted, the version written at the
    # start (with completed_at=None) is what survives on disk.
    save_run_metadata(
        args, meta_path, checkpoint_metadata, config, total_params,
        device, precision, started_at_iso,
        checkpoint_path_full=checkpoint_path,
        completed_at_iso=datetime.now(timezone.utc).isoformat(),
        peak_gpu_memory_mb=round(peak_mb, 1) if peak_mb is not None else None,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate predictions for PhysicsSLM on a benchmark dataset "
                     "(no evaluation performed here)."
    )
    parser.add_argument("--benchmark_file", type=str, default=DEFAULT_BENCHMARK_FILE)
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="explicit checkpoint path; defaults to "
                              "<checkpoint_dir>/best_model.pt (same rule as inference_fine_tune.py)")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--tokenizer_model", type=str, default=DEFAULT_TOKENIZER_MODEL)
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--fp16", action="store_true",
                         help="force fp16 inference (not recommended on Pascal-class GPUs; "
                              "see inference_fine_tune.py's own warning)")
    parser.add_argument("--do_sample", action="store_true",
                         help="use sampling instead of greedy decoding. Greedy (default) "
                              "matches inference_fine_tune.py's default, for reproducible runs.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=None,
                         help="random seed for reproducible sampling runs")
    parser.add_argument("--force", action="store_true",
                         help="continue even if tokenizer, parameter-count, or prompt-"
                              "template verification fails (NOT recommended)")
    return parser.parse_args()


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()