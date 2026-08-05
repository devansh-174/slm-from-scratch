"""
Production Tokenizer Training
===============================
Trains the FINAL SentencePiece tokenizer to actually be used for model
training. Settings are locked to the confirmed production configuration:

    vocab size     : requested at LEARNED_PIECES_TARGET + special/custom
                      tokens, but the FINAL size is whatever the corpus
                      naturally supports -- NO post-hoc padding is applied.
                      Padding to force an exact round number would waste
                      embedding parameters on inert placeholder tokens that
                      are never produced during real encoding, which isn't
                      worth it at this model scale. Whatever the real corpus
                      achieves IS the final, honest vocab size.
    context length : 512   (fixed by explicit choice for this run --
                             informational here, actually enforced later
                             during dataset batching / model config.)
    special tokens : <CALC>, </CALC>  (kept atomic)
    unk token      : exactly ONE dedicated slot (unk_id=0), standard
                      SentencePiece reservation -- no extra unknown tokens.
    split_digits   : True   (each digit its own token -- good for numerics)
    byte_fallback  : True
    lowercasing    : OFF
    normalization  : NFKC

CLEAN_PLACEHOLDERS (default True): strips fill-in-the-blank / truncation
artifacts (runs of 3+ non-alphanumeric characters, e.g. "____", "....",
"----", "___.") out of the corpus BEFORE tokenizer training, collapsing
each one down to a single canonical "___" marker. Content inside
<CALC>...</CALC> is never touched by this cleaning step.

USAGE:
    python3 train_tokenizer.py
"""

import io
import json
import re
from pathlib import Path

import sentencepiece as spm

# ---------- CONFIG (confirmed final settings) ----------
TRAIN_FILE = Path("train.jsonl")
OUTPUT_DIR = Path("tokenizer")
MODEL_PREFIX = "slm_tokenizer"

# Requested target -- the corpus may not support this exactly, and that's
# fine. See docstring above: no padding is applied to force this number.
LEARNED_PIECES_TARGET = 12773
N_SPECIAL_TOKENS = 4  # unk, bos, eos, pad -- always reserved by SentencePiece
SPECIAL_TOKENS = ["<CALC>", "</CALC>"]
VOCAB_SIZE = LEARNED_PIECES_TARGET + N_SPECIAL_TOKENS + len(SPECIAL_TOKENS)

CONTEXT_LENGTH = 512
CHAR_COVERAGE = 1.0
SPLIT_DIGITS = True

CLEAN_PLACEHOLDERS = True
PLACEHOLDER_CANONICAL = "___"

ANSWER_ORDER = ["given", "formula", "substitution", "explanation", "final_answer"]

PLACEHOLDER_RUN_RE = re.compile(r"[^A-Za-z0-9\s]{3,}")
CALC_SPAN_RE = re.compile(r"<CALC>.*?</CALC>", re.DOTALL)


def clean_placeholders(text):
    """Collapse placeholder/junk runs to a single canonical token, without
    touching anything inside <CALC>...</CALC>."""
    protected = []

    def _protect(m):
        protected.append(m.group(0))
        return f"\x00{len(protected) - 1}\x00"

    shielded = CALC_SPAN_RE.sub(_protect, text)

    n_runs = 0
    n_chars = 0

    def _replace(m):
        nonlocal n_runs, n_chars
        n_runs += 1
        n_chars += len(m.group(0))
        return PLACEHOLDER_CANONICAL

    cleaned = PLACEHOLDER_RUN_RE.sub(_replace, shielded)

    def _restore(m):
        return protected[int(m.group(1))]

    cleaned = re.sub(r"\x00(\d+)\x00", _restore, cleaned)
    return cleaned, n_runs, n_chars


def serialize(rec):
    parts = []
    q = rec.get("question")
    if q:
        parts.append(f"Question: {q}")
    ans = rec.get("answer")
    if isinstance(ans, dict):
        for k in ANSWER_ORDER:
            v = ans.get(k)
            if v not in (None, ""):
                parts.append(f"{k.replace('_', ' ').title()}: {v}")
        for k, v in ans.items():
            if k not in ANSWER_ORDER and v not in (None, ""):
                parts.append(f"{k}: {v}")
    elif ans not in (None, ""):
        parts.append(f"Answer: {ans}")
    return "\n".join(parts)


def build_corpus():
    OUTPUT_DIR.mkdir(exist_ok=True)
    corpus_txt = OUTPUT_DIR / "train_corpus.txt"

    print(f"Reading {TRAIN_FILE} and serializing...")
    n_written = 0
    total_runs_cleaned = 0
    total_chars_cleaned = 0
    with io.open(TRAIN_FILE, "r", encoding="utf-8") as fin, \
         io.open(corpus_txt, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            s = serialize(rec)
            if not s.strip():
                continue
            if CLEAN_PLACEHOLDERS:
                s, n_runs, n_chars = clean_placeholders(s)
                total_runs_cleaned += n_runs
                total_chars_cleaned += n_chars
            fout.write(s)
            fout.write("\n\n")
            n_written += 1

    print(f"Wrote {n_written:,} serialized training samples to {corpus_txt}")
    if CLEAN_PLACEHOLDERS:
        print(f"Placeholder cleaning: {total_runs_cleaned:,} runs collapsed to "
              f"'{PLACEHOLDER_CANONICAL}' ({total_chars_cleaned:,} raw characters affected)")
    return corpus_txt, n_written


def train_one(corpus_txt, vocab_size, label):
    requested_vocab_size = vocab_size
    model_path_prefix = str(OUTPUT_DIR / f"{MODEL_PREFIX}_{label}")
    print(f"\nTraining SentencePiece tokenizer -> {model_path_prefix}.model / .vocab")
    print(f"  vocab_size={vocab_size}, split_digits={SPLIT_DIGITS}, "
          f"special_tokens={SPECIAL_TOKENS}, char_coverage={CHAR_COVERAGE}")

    def _train(v):
        spm.SentencePieceTrainer.Train(
            input=str(corpus_txt),
            model_prefix=model_path_prefix,
            model_type="unigram",
            vocab_size=v,
            character_coverage=CHAR_COVERAGE,
            byte_fallback=True,
            normalization_rule_name="nfkc",
            add_dummy_prefix=True,
            remove_extra_whitespaces=False,
            split_digits=SPLIT_DIGITS,
            user_defined_symbols=SPECIAL_TOKENS,
            unk_id=0, bos_id=1, eos_id=2, pad_id=3,
        )

    achieved_vocab_size = requested_vocab_size
    try:
        _train(requested_vocab_size)
    except RuntimeError as e:
        msg = str(e)
        match = re.search(r"<=\s*(\d+)", msg)
        if match:
            max_achievable = int(match.group(1))
            print(f"  Corpus can't support vocab_size={requested_vocab_size} "
                  f"(max achievable is {max_achievable}). Training at {max_achievable} "
                  f"-- this is the FINAL size, no padding will be applied.")
            achieved_vocab_size = max_achievable
            _train(achieved_vocab_size)
        else:
            raise

    print(f"\nDone. Model files:")
    print(f"  {model_path_prefix}.model")
    print(f"  {model_path_prefix}.vocab")

    return model_path_prefix, requested_vocab_size, achieved_vocab_size


def sanity_check(sp):
    print(f"\nFinal vocab size: {sp.GetPieceSize()}")
    probes = [
        "Substitution: <CALC>1 * 5</CALC>",
        "A car travels 60 km in 2 hours. What is its average speed?",
    ]
    print("\nSanity check -- encode/decode round-trips:")
    for p in probes:
        pieces = sp.EncodeAsPieces(p)
        ids = sp.EncodeAsIds(p)
        decoded = sp.DecodeIds(ids)
        match = "OK" if decoded == p else "MISMATCH"
        print(f"  input : {p!r}")
        print(f"  pieces: {pieces}")
        print(f"  decoded round-trip: {match}")
        print()


def main():
    corpus_txt, n_written = build_corpus()

    # Train the single production tokenizer. Whatever the corpus naturally
    # supports IS the final vocab size -- no padding applied.
    model_path_prefix, requested, achieved = train_one(
        corpus_txt, VOCAB_SIZE, str(VOCAB_SIZE)
    )
    if achieved != requested:
        print(f"\nCorpus supports {achieved} pieces (requested {requested}). "
              f"Using the natural achievable size -- no padding applied.")

    sp = spm.SentencePieceProcessor()
    sp.Load(model_path_prefix + ".model")
    sanity_check(sp)

    actual_learned_pieces = achieved - N_SPECIAL_TOKENS - len(SPECIAL_TOKENS)

    print("\n" + "=" * 80)
    print("Production Tokenizer Config (final, no padding)")
    print("=" * 80)
    print(f"  Vocab size (final)     : {achieved:,}")
    print(f"    = {actual_learned_pieces:,} learned pieces "
          f"+ {N_SPECIAL_TOKENS} special (unk={sp.unk_id()}, bos={sp.bos_id()}, "
          f"eos={sp.eos_id()}, pad={sp.pad_id()}) "
          f"+ {len(SPECIAL_TOKENS)} custom ({', '.join(SPECIAL_TOKENS)})")
    print(f"  Context length         : {CONTEXT_LENGTH:,}  (fixed by explicit choice)")
    print(f"  Placeholder cleaning   : {'ON' if CLEAN_PLACEHOLDERS else 'OFF'}")
    print(f"  Model files            : {model_path_prefix}.model / .vocab")
    print("=" * 80)


if __name__ == "__main__":
    main()