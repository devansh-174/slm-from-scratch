"""
Tokenizes train1.jsonl and validation1.jsonl into .bin token-id files.

Uses the tok_analysis_v2 production tokenizer (spm_11000.model).

USAGE:
    python3 make_bins2.py

serialize()/clean_text() below are exact copies from train_tokenizer2.py.
"""

import json
import random
import re
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import sentencepiece as spm

TRAIN_JSONL = Path("train1.jsonl")
VAL_JSONL = Path("validation1.jsonl")

TOKENIZER_MODEL = Path("tok_analysis_v2/spm_11000.model")

TRAIN_BIN = Path("train2.bin")
VAL_BIN = Path("val2.bin")
STATS_PATH = Path("bin_stats.json")

VAL_FRACTION = 0.10
SEED = 42

CONTEXT_WARNING_THRESHOLD = 2048


INVISIBLE_CHARS = "\u200b\u200c\u200d\ufeff\u200e\u200f\u2060\u00ad"
_invisible_re = re.compile(f"[{re.escape(INVISIBLE_CHARS)}]")
_whitespace_re = re.compile(r"[ \t]+")  # collapse repeated spaces/tabs only


def clean_text(s):
    """
    Strips invisible/zero-width Unicode characters (zero-width space/joiner/
    non-joiner, BOM, LTR/RTL marks, word joiner, soft hyphen) that silently
    create useless vocabulary entries, and collapses repeated spaces/tabs
    (common in OCR'd / PDF-sourced text) while preserving newlines so the
    Question/Options/Answer structure stays intact.

    (Exact copy from train_tokenizer2.py -- must match the tokenizer's
    training-time text processing exactly.)
    """
    s = _invisible_re.sub("", s)
    s = _whitespace_re.sub(" ", s)
    return s


def serialize(rec):
    """
    Convert one JSON record into text for tokenization.

    (Exact copy of train_tokenizer2.py's serialize(). Answer and Solution are
    merged into a single "Answer:" section: the answer sits on its own line
    right after "Answer:", and the explanation -- when present -- follows as
    a blank-line-separated paragraph underneath. There is no separate
    "Solution:" header.)
    """
    parts = []

    question = rec.get("question")
    if question:
        parts.append(f"Question:\n{question}")

    options = rec.get("options")
    if isinstance(options, dict) and options:
        opt_lines = ["Options:"]
        for key in sorted(options.keys()):
            value = options[key]
            if value not in (None, ""):
                opt_lines.append(f"{key}) {value}")
        parts.append("\n".join(opt_lines))

    answer = rec.get("answer")
    solution = rec.get("solution")
    if answer not in (None, ""):
        answer_block = f"Answer:\n{answer}"
        if solution not in (None, ""):
            answer_block += f"\n\n{solution}"
        parts.append(answer_block)
    elif solution not in (None, ""):
        parts.append(f"Answer:\n{solution}")

    return clean_text("\n\n".join(parts))


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def split_train_val(rows, val_fraction, seed):
    """
    Only used if validation file does not exist.
    """

    if rows and "source_concept_id" in rows[0]:

        by_concept = defaultdict(list)

        for r in rows:
            by_concept[r["source_concept_id"]].append(r)

        concept_ids = list(by_concept.keys())

        random.Random(seed).shuffle(concept_ids)

        val_count = max(1, int(len(concept_ids) * val_fraction))

        val_ids = set(concept_ids[:val_count])

        train_rows = []
        val_rows = []

        for cid, group in by_concept.items():

            if cid in val_ids:
                val_rows.extend(group)
            else:
                train_rows.extend(group)

        print(
            f"Split by concept: "
            f"{len(train_rows):,} train rows | "
            f"{len(val_rows):,} val rows"
        )

    else:

        rows = rows[:]

        random.Random(seed).shuffle(rows)

        val_count = max(1, int(len(rows) * val_fraction))

        val_rows = rows[:val_count]
        train_rows = rows[val_count:]

    return train_rows, val_rows


def tokenize_rows(rows, sp, stats):
    """
    stats: dict accumulating {"max_tokens": int, "over_threshold": int}
    """

    eos = sp.eos_id()

    ids = []

    for row in rows:

        text = serialize(row)

        if not text.strip():
            continue

        token_ids = sp.EncodeAsIds(text)

        assert len(token_ids) > 0, "Tokenizer produced zero tokens for a non-empty text"
        assert max(token_ids) < sp.GetPieceSize(), (
            f"Token id {max(token_ids)} out of range for vocab_size={sp.GetPieceSize()}"
        )
        assert min(token_ids) >= 0, f"Negative token id encountered: {min(token_ids)}"

        stats["unk_count"] += token_ids.count(sp.unk_id())

        stats["max_tokens"] = max(stats["max_tokens"], len(token_ids))
        if len(token_ids) > CONTEXT_WARNING_THRESHOLD:
            stats["over_threshold"] += 1

        ids.extend(token_ids)
        ids.append(eos)

    return ids


def save_bin(ids, path):

    arr = np.array(ids, dtype=np.uint16)

    arr.tofile(path)

    print(
        f"{path.name:<12}"
        f"{len(arr):>15,} tokens"
        f"    {path.stat().st_size/1024/1024:.2f} MB"
    )


def main():

    print("=" * 60)

    print("Loading tokenizer...")

    sp = spm.SentencePieceProcessor()

    if not sp.Load(str(TOKENIZER_MODEL)):
        raise RuntimeError(f"Could not load tokenizer: {TOKENIZER_MODEL}")

    print("Vocabulary:", sp.GetPieceSize())

    assert sp.GetPieceSize() == 11000, (
        f"Expected vocab_size=11000, got {sp.GetPieceSize()} — "
        f"tokenizer/model mismatch, do not proceed."
    )

    assert sp.unk_id() == 0, f"Expected unk_id=0, got {sp.unk_id()}"
    assert sp.bos_id() == 1, f"Expected bos_id=1, got {sp.bos_id()}"
    assert sp.eos_id() == 2, f"Expected eos_id=2, got {sp.eos_id()}"
    assert sp.pad_id() == 3, f"Expected pad_id=3, got {sp.pad_id()}"

    print("=" * 60)

    print("Loading training data...")

    train_rows = load_jsonl(TRAIN_JSONL)

    print(f"Train rows : {len(train_rows):,}")

    if VAL_JSONL.exists():

        val_rows = load_jsonl(VAL_JSONL)

        print(f"Val rows   : {len(val_rows):,}")

    else:

        train_rows, val_rows = split_train_val(
            train_rows,
            VAL_FRACTION,
            SEED,
        )

    print("=" * 60)

    print("Tokenizing training set...")

    train_stats = {"max_tokens": 0, "over_threshold": 0, "unk_count": 0}
    train_ids = tokenize_rows(train_rows, sp, train_stats)

    print(f"Train tokens : {len(train_ids):,}")
    print(f"Longest sample: {train_stats['max_tokens']} tokens")
    print(f"Samples >{CONTEXT_WARNING_THRESHOLD} tokens: {train_stats['over_threshold']}")

    print()

    print("Tokenizing validation set...")

    val_stats = {"max_tokens": 0, "over_threshold": 0, "unk_count": 0}
    val_ids = tokenize_rows(val_rows, sp, val_stats)

    print(f"Validation tokens : {len(val_ids):,}")
    print(f"Longest sample: {val_stats['max_tokens']} tokens")
    print(f"Samples >{CONTEXT_WARNING_THRESHOLD} tokens: {val_stats['over_threshold']}")

    print()

    total = len(train_ids) + len(val_ids)
    overall_max_tokens = max(train_stats["max_tokens"], val_stats["max_tokens"])
    overall_over_threshold = train_stats["over_threshold"] + val_stats["over_threshold"]

    avg_train = len(train_ids) / len(train_rows) if train_rows else 0
    avg_val = len(val_ids) / len(val_rows) if val_rows else 0
    print(f"Average tokens/sample -- train: {avg_train:.1f}  val: {avg_val:.1f}")

    print(f"TOTAL TOKENS : {total:,}")
    print(f"OVERALL Longest sample: {overall_max_tokens} tokens")
    print(f"OVERALL Samples >{CONTEXT_WARNING_THRESHOLD} tokens: {overall_over_threshold}")

    print("=" * 60)

    print("Saving binaries...")

    save_bin(train_ids, TRAIN_BIN)

    save_bin(val_ids, VAL_BIN)

    tokenizer_sha256 = hashlib.sha256(TOKENIZER_MODEL.read_bytes()).hexdigest()
    tokenizer_filename = TOKENIZER_MODEL.name

    stats = {
        "total_train_tokens": len(train_ids),
        "total_val_tokens": len(val_ids),
        "total_tokens": total,
        "avg_tokens_per_sample_train": avg_train,
        "avg_tokens_per_sample_val": avg_val,
        "longest_sample_train": train_stats["max_tokens"],
        "longest_sample_val": val_stats["max_tokens"],
        "longest_sample_overall": overall_max_tokens,
        "samples_over_2048_train": train_stats["over_threshold"],
        "samples_over_2048_val": val_stats["over_threshold"],
        "samples_over_2048_overall": overall_over_threshold,
        "unk_tokens": train_stats["unk_count"] + val_stats["unk_count"],
        "tokenizer_path": str(TOKENIZER_MODEL),
        "tokenizer_file": tokenizer_filename,
        "tokenizer_sha256": tokenizer_sha256,
        "vocab_size": sp.GetPieceSize(),
        "train_bin": str(TRAIN_BIN),
        "val_bin": str(VAL_BIN),
    }

    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved stats -> {STATS_PATH}")

    print("=" * 60)

    print("Finished successfully.")

    print()

    print("Use these with train1.py:")

    print(f"--train_bin {TRAIN_BIN}")

    print(f"--val_bin {VAL_BIN}")


if __name__ == "__main__":
    main()