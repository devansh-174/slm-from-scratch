"""
inference_fine_tune.py

Interactive inference / evaluation script for the supervised fine-tuned 300M
Physics Small Language Model.

Architecture, tokenizer, and prompt serialization are NOT redefined here —
they are imported / matched exactly from model2.py and finetune.py so that
Base-vs-SFT comparisons are meaningful. See the header comment blocks below
for the specific contract each piece must satisfy.

Run from the project root, e.g.:

    python3 inference_fine_tune.py
    python3 inference_fine_tune.py --checkpoint finetune_checkpoints/best_model.pt
    CUDA_VISIBLE_DEVICES=1 python3 inference_fine_tune.py
"""

import os
import sys
import glob
import time
import argparse
import math

import torch

from model2 import PhysicsSLM, PhysicsSLMConfig

try:
    import sentencepiece as spm
except ImportError:
    spm = None


EXPECTED_PARAM_COUNT = 300_043_776
EXPECTED_VOCAB_SIZE = 11000
EXPECTED_UNK_ID = 0
EXPECTED_BOS_ID = 1
EXPECTED_EOS_ID = 2
EXPECTED_PAD_ID = 3
MAX_CONTEXT = 2048

# This MUST match format_sample()'s prompt_part in finetune.py exactly.
# Verified directly against finetune.py's source (not assumed): it builds
#   prompt_part = f"<QUESTION>\n\n{instruction}\n\n{ANSWER_TAG}\n\n"
# with ANSWER_TAG = "<ANSWER>", and the same literal template is used a
# third time in finetune.py's own run_generation_and_save() eval-prompt
# construction, so all three sites agree. Reproduced verbatim here rather
# than re-derived, so a future drift in finetune.py's template is something
# you'll notice (mismatched outputs) rather than something silently invisible.
ANSWER_TAG = "<ANSWER>"


def build_sft_prompt(instruction: str) -> str:
    """EXACT prompt serialization used during fine-tuning. Do not modify
    without also updating finetune.py's format_sample(), or inference will
    silently diverge from what the model was actually trained on."""
    return f"<QUESTION>\n\n{instruction.strip()}\n\n{ANSWER_TAG}\n\n"


# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------
def load_tokenizer(tokenizer_path: str):
    if spm is None:
        print("ERROR: sentencepiece is not installed. `pip install sentencepiece`.")
        sys.exit(1)
    if not os.path.exists(tokenizer_path):
        print(f"ERROR: tokenizer model not found at: {tokenizer_path}")
        sys.exit(1)

    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)

    errors = []
    if sp.GetPieceSize() != EXPECTED_VOCAB_SIZE:
        errors.append(f"vocab size {sp.GetPieceSize()} != expected {EXPECTED_VOCAB_SIZE}")
    if sp.unk_id() != EXPECTED_UNK_ID:
        errors.append(f"unk_id {sp.unk_id()} != expected {EXPECTED_UNK_ID}")
    if sp.bos_id() != EXPECTED_BOS_ID:
        errors.append(f"bos_id {sp.bos_id()} != expected {EXPECTED_BOS_ID}")
    if sp.eos_id() != EXPECTED_EOS_ID:
        errors.append(f"eos_id {sp.eos_id()} != expected {EXPECTED_EOS_ID}")
    if sp.pad_id() != EXPECTED_PAD_ID:
        errors.append(f"pad_id {sp.pad_id()} != expected {EXPECTED_PAD_ID}")

    if errors:
        print("ERROR: tokenizer does not match the expected fine-tuning configuration:")
        for e in errors:
            print(f"  - {e}")
        print("Refusing to continue with a mismatched tokenizer.")
        sys.exit(1)

    return sp


# --------------------------------------------------------------------------
# Checkpoint resolution / loading
# --------------------------------------------------------------------------
def resolve_checkpoint_path(explicit_path: str, checkpoint_dir: str) -> str:
    """
    Default is the BEST fine-tuned checkpoint produced by finetune.py, which
    writes it to <out_dir>/best_model.pt. We do NOT fall back to
    latest_model.pt or any other checkpoint if this is missing — that would
    silently change which model is being evaluated.
    """
    if explicit_path:
        if not os.path.exists(explicit_path):
            print(f"ERROR: specified --checkpoint does not exist: {explicit_path}")
            sys.exit(1)
        return explicit_path

    default_path = os.path.join(checkpoint_dir, "best_model.pt")
    if os.path.exists(default_path):
        return default_path

    print(f"ERROR: default best checkpoint not found: {default_path}")
    available = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt")))
    if available:
        print(f"Available .pt files in {checkpoint_dir}/:")
        for p in available:
            print(f"  - {p}")
    else:
        print(f"No .pt files found in {checkpoint_dir}/ at all.")
    print("Pass an explicit path with --checkpoint if you want to load a different one. "
          "Refusing to silently fall back to another checkpoint.")
    sys.exit(1)


def strip_module_prefix(state_dict: dict) -> dict:
    """Removes a 'module.' prefix left over from DataParallel-saved checkpoints,
    only when present — leaves non-DataParallel state dicts untouched."""
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):] if k.startswith("module.") else k: v
                for k, v in state_dict.items()}
    return state_dict


def load_model_from_checkpoint(checkpoint_path: str, device: str):
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"ERROR: failed to load checkpoint file: {e}")
        sys.exit(1)

    if "model_state_dict" not in ckpt:
        print("ERROR: checkpoint does not contain a 'model_state_dict' key; "
              "this does not look like a finetune.py checkpoint.")
        sys.exit(1)

    config = PhysicsSLMConfig()  # locked architecture — DO NOT modify
    model = PhysicsSLM(config)

    state_dict = strip_module_prefix(ckpt["model_state_dict"])

    try:
        # strict=True: any missing or unexpected key is a hard failure, not
        # a warning. A checkpoint that doesn't exactly match the 300M
        # architecture must not load "mostly fine".
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print("ERROR: checkpoint state_dict does not match the instantiated "
              "300M architecture (strict=True load failed).")
        print(str(e))
        sys.exit(1)

    total_params = sum(p.numel() for p in model.parameters())
    if total_params != EXPECTED_PARAM_COUNT:
        print(f"ERROR: loaded model has {total_params:,} parameters, "
              f"expected exactly {EXPECTED_PARAM_COUNT:,}.")
        sys.exit(1)

    model.eval()
    model.to(device)

    metadata = {
        "checkpoint_filename": os.path.basename(checkpoint_path),
        "step": ckpt.get("step", "N/A"),
        "epoch": ckpt.get("epoch", "N/A"),
        "best_val_loss": ckpt.get("best_val_loss", "N/A"),
        "val_loss": ckpt.get("val_loss", "N/A"),
        "patience_counter": ckpt.get("patience_counter", "N/A"),
    }
    return model, config, metadata, total_params


# --------------------------------------------------------------------------
# Device / precision
# --------------------------------------------------------------------------
# Pascal = compute capability 6.x. This is the generation that lacks fast
# fp16 tensor-core throughput (Volta/sm_70+ is where that shows up), so the
# specific "don't bother with fp16 here" warning is only accurate for
# Pascal-class cards. We gate the message on the actual queried compute
# capability instead of hardcoding a single GPU name, so this doesn't print
# a misleading warning on unrelated hardware (e.g. an A100 or a 4090).
PASCAL_COMPUTE_CAPABILITY_MAJOR = 6


def setup_device_and_precision(args):
    if torch.cuda.is_available():
        device = "cuda:0"
        gpu_name = torch.cuda.get_device_properties(0).name
        gpu_count_visible = torch.cuda.device_count()
        cc_major, cc_minor = torch.cuda.get_device_capability(0)
        print("=" * 60)
        print(f"Device                 : {device}")
        print(f"GPU name               : {gpu_name}")
        print(f"GPU count visible      : {gpu_count_visible}")
        print(f"CUDA compute capability: sm_{cc_major}{cc_minor}")
        print("=" * 60)
    else:
        device = "cpu"
        gpu_name = None
        cc_major, cc_minor = None, None
        print("=" * 60)
        print("Device                 : cpu (CUDA not available)")
        print("=" * 60)

    if args.fp16:
        if device == "cpu":
            print("WARNING: --fp16 requested but CUDA is unavailable; falling back to fp32 on CPU.")
            precision = "fp32"
        elif cc_major == PASCAL_COMPUTE_CAPABILITY_MAJOR:
            print(f"WARNING: --fp16 forced on a Pascal GPU ({gpu_name}, sm_{cc_major}{cc_minor}). "
                  "Pascal lacks fast fp16 tensor-core throughput, so fp16 here may not give the "
                  "speed/quality tradeoffs you'd see on newer (Volta+) architectures. Provided for "
                  "explicit opt-in only.")
            precision = "fp16"
        else:
            print(f"NOTE: --fp16 forced on {gpu_name} (sm_{cc_major}{cc_minor}).")
            precision = "fp16"
    else:
        precision = "fp32"

    return device, precision


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
@torch.inference_mode()
def generate_response(model, tokenizer, instruction, device, precision,
                       do_sample, temperature, top_p, top_k, max_new_tokens,
                       repetition_penalty):
    prompt_text = build_sft_prompt(instruction)

    t_encode_start = time.time()
    prompt_ids = tokenizer.encode(prompt_text, out_type=int)
    t_encode_end = time.time()

    prompt_len = len(prompt_ids)
    if prompt_len >= MAX_CONTEXT:
        print(f"ERROR: prompt is {prompt_len} tokens, which already meets or exceeds the "
              f"{MAX_CONTEXT}-token context window. Cannot generate. Shorten the question.")
        return None

    # Dynamically reduce max_new_tokens so prompt_tokens + max_new_tokens <= 2048.
    allowed_new_tokens = MAX_CONTEXT - prompt_len
    effective_max_new_tokens = min(max_new_tokens, allowed_new_tokens)
    if effective_max_new_tokens < max_new_tokens:
        print(f"NOTE: reducing max_new_tokens from {max_new_tokens} to {effective_max_new_tokens} "
              f"to respect the {MAX_CONTEXT}-token context window (prompt uses {prompt_len} tokens).")

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                     if precision == "fp16" and device.startswith("cuda")
                     else torch.autocast(device_type="cpu", enabled=False))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_gen_start = time.time()

    try:
        with autocast_ctx:
            generated = model.generate(
                input_ids,
                max_new_tokens=effective_max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=EXPECTED_EOS_ID,
                do_sample=do_sample,
            )
    except Exception as e:
        print(f"ERROR during generation: {e}")
        return None

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_gen_end = time.time()

    # Guard against a model that has silently produced NaN/Inf (e.g. an
    # fp16 overflow). `generated` here is a token-id tensor, not logits, so
    # this can only catch it indirectly: a NaN/Inf in the underlying logits
    # will not itself show up as a NaN id (ids are integers), but some
    # generate() implementations propagate a sentinel/negative id or raise
    # internally when this happens. We check for that here rather than
    # silently decoding whatever came out. If model2.py's generate() exposes
    # per-step logits in the future, prefer checking those directly with
    # torch.isnan(logits).any() / torch.isinf(logits).any() instead of this
    # id-level check.
    if not torch.isfinite(generated.float()).all():
        print("ERROR: generation produced non-finite token ids (NaN/Inf). "
              "This usually indicates numerical overflow (common with forced fp16 "
              "on unsupported hardware). Discarding this output rather than decoding garbage.")
        return None

    # Only decode newly generated tokens — never the prompt.
    new_tokens = generated[0][input_ids.shape[1]:].tolist()

    # Stop cleanly at EOS: truncate anything at/after the first EOS token,
    # since model.generate() may still emit a few more tokens in the same
    # batched call before the caller-level loop notices.
    if EXPECTED_EOS_ID in new_tokens:
        new_tokens = new_tokens[:new_tokens.index(EXPECTED_EOS_ID)]

    t_decode_start = time.time()
    # Do not print special tokens (EOS/PAD/BOS); SentencePiece's decode()
    # already omits control pieces for a normal vocab, but we've also
    # explicitly excluded EOS above.
    text = tokenizer.decode(new_tokens)
    text = text.strip()  # remove only accidental trailing/leading whitespace
    t_decode_end = time.time()

    timings = {
        "encode_ms": (t_encode_end - t_encode_start) * 1000,
        "generation_ms": (t_gen_end - t_gen_start) * 1000,
        "decode_ms": (t_decode_end - t_decode_start) * 1000,
    }
    timings["total_ms"] = timings["encode_ms"] + timings["generation_ms"] + timings["decode_ms"]

    stats = {
        "prompt_tokens": prompt_len,
        "generated_tokens": len(new_tokens),
        "total_tokens": prompt_len + len(new_tokens),
    }

    return text, timings, stats


# --------------------------------------------------------------------------
# Interactive interface
# --------------------------------------------------------------------------
HELP_TEXT = """
Available commands:
  /help                 Show this help message
  /greedy               Switch to deterministic greedy decoding
  /sample                Switch to sampling (temperature/top_p/top_k)
  /temp <value>          Set sampling temperature (> 0)
  /top_p <value>         Set sampling top_p (0, 1]
  /top_k <value>         Set sampling top_k (>= 0; 0 disables top_k)
  /max_tokens <int>      Set max_new_tokens (> 0)
  /info                  Show checkpoint / architecture / decoding info
  /clear                 Clear CUDA cache
  /quit, /exit           Exit the program

Anything else you type is treated as a physics question.
""".strip()


def print_info(metadata, config, total_params, device, precision, do_sample,
                temperature, top_p, top_k, max_new_tokens, repetition_penalty):
    print("-" * 60)
    print(f"Checkpoint          : {metadata['checkpoint_filename']}")
    print(f"Step                : {metadata['step']}")
    print(f"Epoch               : {metadata['epoch']}")
    print(f"Best val loss       : {metadata['best_val_loss']}")
    print(f"Val loss            : {metadata['val_loss']}")
    print(f"Patience counter    : {metadata['patience_counter']}")
    print(f"Architecture        : {config.n_layers}L / hidden={config.hidden_size} / "
          f"heads={config.n_heads} / kv_heads={config.n_kv_heads}")
    print(f"Parameters          : {total_params:,}")
    print(f"Context length      : {config.max_position_embeddings}")
    print(f"Device              : {device}")
    print(f"Precision           : {precision.upper()}")
    print(f"Decoding            : {'SAMPLE' if do_sample else 'GREEDY'}")
    if do_sample:
        print(f"  temperature={temperature}  top_p={top_p}  top_k={top_k}")
    print(f"Repetition penalty  : {repetition_penalty}")
    print(f"Max new tokens      : {max_new_tokens}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="explicit checkpoint path; defaults to finetune_checkpoints/best_model.pt")
    parser.add_argument("--checkpoint_dir", type=str, default="finetune_checkpoints")
    parser.add_argument("--tokenizer_model", type=str, default="tok_analysis_v2/spm_11000.model")
    parser.add_argument("--fp16", action="store_true",
                         help="force fp16 inference (not recommended on Pascal-class GPUs)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--repetition_penalty", type=float, default=1.05,
                         help="passed straight to model.generate()'s repetition_penalty arg, "
                              "which model2.py's PhysicsSLM.generate() supports natively")
    args = parser.parse_args()

    # ---------------- Tokenizer ----------------
    tokenizer = load_tokenizer(args.tokenizer_model)

    # ---------------- Device / precision ----------------
    device, precision = setup_device_and_precision(args)

    # ---------------- Checkpoint ----------------
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.checkpoint_dir)
    model, config, metadata, total_params = load_model_from_checkpoint(checkpoint_path, device)

    # ---------------- Decoding state (mutable via interactive commands) --------
    do_sample = False  # greedy by default, per spec (reproducible Base-vs-SFT comparisons)
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    max_new_tokens = args.max_new_tokens
    repetition_penalty = args.repetition_penalty

    # ---------------- Startup header ----------------
    print("=" * 45)
    print("Fine-Tuned 300M Physics SLM")
    print("=" * 45)
    print(f"Tokenizer        : {os.path.basename(args.tokenizer_model)}")
    print(f"Vocabulary       : {tokenizer.GetPieceSize()}")
    print(f"Parameters       : {total_params:,}")
    print(f"Layers           : {config.n_layers}")
    print(f"Hidden size      : {config.hidden_size}")
    print(f"Heads            : {config.n_heads}")
    print(f"KV Heads         : {config.n_kv_heads}")
    print(f"Context          : {config.max_position_embeddings}")
    print(f"Checkpoint       : {metadata['checkpoint_filename']}")
    print(f"Checkpoint step  : {metadata['step']}")
    print(f"Checkpoint epoch : {metadata['epoch']}")
    print(f"Best val loss    : {metadata['best_val_loss']}")
    print(f"Device           : {device}")
    print(f"Precision        : {precision.upper()}")
    print(f"Decoding         : {'GREEDY' if not do_sample else 'SAMPLE'}")
    print(f"Max new tokens   : {max_new_tokens}")
    print("=" * 45)
    print("Type /help for commands, or type a physics question.")

    # ---------------- Interactive loop ----------------
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else None

            if cmd in ("/quit", "/exit"):
                print("Exiting.")
                break

            elif cmd == "/help":
                print(HELP_TEXT)

            elif cmd == "/greedy":
                do_sample = False
                print("Decoding mode: GREEDY")

            elif cmd == "/sample":
                do_sample = True
                print("Decoding mode: SAMPLE "
                      f"(temperature={temperature}, top_p={top_p}, top_k={top_k})")

            elif cmd == "/temp":
                if arg is None:
                    print("Usage: /temp <value>")
                    continue
                try:
                    value = float(arg)
                except ValueError:
                    print(f"Invalid value for /temp: {arg!r}")
                    continue
                if not math.isfinite(value) or value <= 0:
                    print(f"Invalid value for /temp: {arg!r} (must be a finite number > 0)")
                    continue
                temperature = value
                print(f"temperature set to {temperature}")

            elif cmd == "/top_p":
                if arg is None:
                    print("Usage: /top_p <value>")
                    continue
                try:
                    value = float(arg)
                except ValueError:
                    print(f"Invalid value for /top_p: {arg!r}")
                    continue
                if not math.isfinite(value) or not (0 < value <= 1):
                    print(f"Invalid value for /top_p: {arg!r} (must be in the range (0, 1])")
                    continue
                top_p = value
                print(f"top_p set to {top_p}")

            elif cmd == "/top_k":
                if arg is None:
                    print("Usage: /top_k <value>")
                    continue
                try:
                    value = int(arg)
                except ValueError:
                    print(f"Invalid value for /top_k: {arg!r}")
                    continue
                if value < 0:
                    print(f"Invalid value for /top_k: {arg!r} (must be >= 0; 0 disables top_k)")
                    continue
                top_k = value
                print(f"top_k set to {top_k}")

            elif cmd == "/max_tokens":
                if arg is None:
                    print("Usage: /max_tokens <integer>")
                    continue
                try:
                    value = int(arg)
                except ValueError:
                    print(f"Invalid value for /max_tokens: {arg!r}")
                    continue
                if value <= 0:
                    print(f"Invalid value for /max_tokens: {arg!r} (must be > 0)")
                    continue
                if value >= MAX_CONTEXT:
                    print(f"NOTE: {value} exceeds the {MAX_CONTEXT}-token context window; "
                          f"it will be reduced per-prompt at generation time.")
                max_new_tokens = value
                print(f"max_new_tokens set to {max_new_tokens}")

            elif cmd == "/info":
                print_info(metadata, config, total_params, device, precision, do_sample,
                           temperature, top_p, top_k, max_new_tokens, repetition_penalty)

            elif cmd == "/clear":
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("CUDA cache cleared.")
                else:
                    print("CUDA is not available; nothing to clear.")

            else:
                print(f"Unknown command: {cmd}. Type /help for a list of commands.")

            continue

        # ---------------- Treat input as a physics question ----------------
        result = generate_response(
            model, tokenizer, user_input, device, precision,
            do_sample, temperature, top_p, top_k, max_new_tokens, repetition_penalty,
        )
        if result is None:
            continue
        text, timings, stats = result

        print(f"\n{text}")

        print("\n--- Timing ---")
        print(f"Encode      : {timings['encode_ms']:.1f} ms")
        print(f"Generation  : {timings['generation_ms']:.1f} ms")
        print(f"Decode      : {timings['decode_ms']:.1f} ms")
        print(f"Total       : {timings['total_ms']:.1f} ms")

        print("\n--- Token statistics ---")
        print(f"Prompt tokens    : {stats['prompt_tokens']}")
        print(f"Generated tokens : {stats['generated_tokens']}")
        print(f"Total            : {stats['total_tokens']}")
        print(f"Context usage    : {stats['total_tokens']} / {MAX_CONTEXT} tokens")


if __name__ == "__main__":
    main()
