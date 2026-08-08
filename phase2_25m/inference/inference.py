"""
inference1.py — minimal, single-purpose inference script for PhysicsSLM.

Workflow (exactly as planned):
    Load tokenizer -> Load checkpoint -> Rebuild model -> User enters
    question -> Tokenize -> Generate -> Decode -> Print answer.

USAGE:
    python3 inference1.py
    python3 inference1.py --checkpoint checkpoints/best_model.pt
    python3 inference1.py --checkpoint checkpoints/latest_model.pt
    python3 inference1.py --temperature 0.2 --top-k 50
    python3 inference1.py --question "Why does ice float on water?"   # single-shot, no REPL
    python3 inference1.py --log session_log.jsonl                      # save every Q&A to disk
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import sentencepiece as spm

from model1 import PhysicsSLM


def load_tokenizer(path):
    if not Path(path).exists():
        sys.exit(f"ERROR: tokenizer not found at {path!r}. "
                  f"Check the path or pass --tokenizer explicitly.")
    sp = spm.SentencePieceProcessor()
    sp.Load(path)
    return sp


def cfg_get(config, key):
    """Config may be saved as either a dataclass/object OR a plain dict
    (e.g. if a checkpoint was saved via vars(model_config)) -- handle both
    so this script doesn't silently break depending on how a given
    checkpoint happened to serialize its config."""
    if isinstance(config, dict):
        return config[key]
    return getattr(config, key)


def load_model(checkpoint_path, device):
    """Checkpoint stores its own config (saved by train.py's save_checkpoint),
    so the model architecture is rebuilt EXACTLY as it was trained -- no risk
    of accidentally using a differe nt n_layer/n_embd than what these weights
    actually are."""
    if not Path(checkpoint_path).exists():
        sys.exit(f"ERROR: checkpoint not found at {checkpoint_path!r}. "
                  f"Common options: checkpoints/best_model.pt, checkpoints/latest_model.pt")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    model = PhysicsSLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)  # not forcing .float() -- stays future-proof if
                               # a later checkpoint is trained/saved in fp16/bf16
    model.eval()

    step = ckpt.get("step")
    best_loss = ckpt.get("best_val_loss")
    print("=" * 56)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Step       : {step}")
    print(f"Best loss  : {best_loss:.4f}" if best_loss is not None else "Best loss  : n/a")
    print("=" * 56)
    checkpoint_meta = {"step": step, "best_val_loss": best_loss}
    return model, config, checkpoint_meta


def clean_runaway_generation(text):
    """Small/undertrained models sometimes don't reliably emit EOS and instead
    keep going, hallucinating a NEW 'Question:' block after answering. Trim
    at the first sign of that so the printed answer doesn't bleed into a
    fabricated follow-up question."""
    for marker in ("\nQuestion:", "\n\nQuestion:"):
        idx = text.find(marker)
        if idx != -1:
            return text[:idx].rstrip()
    return text


def answer_question(model, sp, question, config, device, max_new_tokens=150,
                     temperature=0.7, top_k=100, top_p=0.9):
    # Match the EXACT serialization format the model was trained on
    # (see serialize() in train_tokenizer.py / make_bins.py) so the model
    # sees a prompt shaped like its training data, not something novel.
    prompt = f"Question: {question}\n\nAnswer:"

    input_ids = sp.EncodeAsIds(prompt)
    print(f"Input tokens: {len(input_ids)}")

    # Safety: don't let prompt + generation exceed the model's context window
    max_context = cfg_get(config, "max_position_embeddings")
    if len(input_ids) >= max_context:
        print(f"  WARNING: prompt alone is {len(input_ids)} tokens, at/over the "
              f"{max_context}-token context limit -- truncating from the left.")
        input_ids = input_ids[-(max_context - 1):]

    room_left = max_context - len(input_ids)
    actual_max_new = min(max_new_tokens, max(1, room_left))

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # IMPORTANT: temperature <= 0 must mean true greedy decoding, not a literal
    # division by (near) zero. We don't rely on a "do_sample" kwarg here, since
    # model1.py's generate() signature isn't guaranteed to support one -- instead
    # we force top_k=1, which is mathematically identical to greedy decoding
    # (always picks the single highest-probability token) and works with the
    # SAME generate() signature regardless of temperature. This avoids any risk
    # of a TypeError from an unsupported kwarg while still getting the exact
    # correct behavior.
    is_greedy = temperature <= 0
    effective_temperature = 1.0 if is_greedy else temperature  # value is irrelevant once top_k=1
    effective_top_k = 1 if is_greedy else top_k
    if is_greedy:
        print("  temperature <= 0 -> using greedy decoding (top_k=1)")

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=actual_max_new,
            temperature=effective_temperature,
            top_k=effective_top_k,
            top_p=top_p,
            eos_token_id=sp.eos_id(),
        )
    elapsed = time.time() - start

    generated_tokens = len(output_ids[0]) - len(input_ids)
    tok_per_sec = generated_tokens / max(elapsed, 1e-8)
    print(f"Generated {generated_tokens} tokens "
          f"in {elapsed:.2f}s "
          f"({tok_per_sec:.1f} tok/s)")

    full_text = sp.DecodeIds(output_ids[0].tolist())
    generated_only = sp.DecodeIds(output_ids[0, len(input_ids):].tolist())
    generated_only = clean_runaway_generation(generated_only)

    return full_text, generated_only, prompt, elapsed, generated_tokens


def append_log(log_path, record):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="tokenizer/slm_tokenizer_11000.model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=None,
                         help="Fix the random seed for reproducible sampling -- useful when "
                              "comparing two checkpoints/settings on the SAME question fairly.")
    parser.add_argument("--question", type=str, default=None,
                         help="Ask a single question and exit (no REPL) -- handy for scripting, "
                              "e.g. looping this over a file of probe questions.")
    parser.add_argument("--log", type=str, default=None,
                         help="Append every Q&A (with timing/settings) to this .jsonl file, for "
                              "later comparison across checkpoints/hyperparameters.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    print(f"\nLoading tokenizer from {args.tokenizer} ...")
    sp = load_tokenizer(args.tokenizer)
    print(f"  vocab_size={sp.GetPieceSize()}")

    print(f"\nLoading model from {args.checkpoint} ...")
    model, config, checkpoint_meta = load_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e6:.1f}M parameters, max_context={cfg_get(config, 'max_position_embeddings')}")

    def run_one(question):
        full_text, generated_only, prompt, elapsed, n_tok = answer_question(
            model, sp, question, config, device, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        )
        print(f"\nQuestion: {question}\n\nAnswer: {generated_only.strip()}")

        if args.log:
            append_log(args.log, {
                "checkpoint": args.checkpoint,
                "step": checkpoint_meta["step"],
                "best_val_loss": checkpoint_meta["best_val_loss"],
                "question": question,
                "answer": generated_only.strip(),
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "seed": args.seed,
                "generated_tokens": n_tok,
                "elapsed_sec": round(elapsed, 3),
            })

    # Single-shot mode: answer one question and exit -- no REPL, scriptable.
    if args.question is not None:
        run_one(args.question)
        return

    print("\n" + "=" * 70)
    print("Ready. Type a question (or 'quit' to exit).")
    print("=" * 70)

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        try:
            run_one(question)
        except KeyboardInterrupt:
            print("\n  Generation interrupted -- ready for next question.")
            continue


if __name__ == "__main__":
    main()