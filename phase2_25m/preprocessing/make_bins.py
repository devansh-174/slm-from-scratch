"""
Tokenizes train1.jsonl and validation1.jsonl into .bin token-id files.


USAGE:
    python3 make_bins1.py
"""

import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import sentencepiece as spm

TRAIN_JSONL = Path("train1.jsonl")
VAL_JSONL = Path("validation1.jsonl")

TOKENIZER_MODEL = Path("tokenizer/slm_tokenizer_11000.model")

TRAIN_BIN = Path("train1.bin")
VAL_BIN = Path("val1.bin")

VAL_FRACTION = 0.10
SEED = 42


def serialize(rec):
    """
    Convert one JSON record into text for tokenization.
    """

    parts = []

    question = rec.get("question")
    if question:
        parts.append(f"Question: {question}")

    options = rec.get("options")
    if isinstance(options, dict) and options:
        parts.append("Options:")
        for key in sorted(options.keys()):
            parts.append(f"{key}. {options[key]}")

    answer = rec.get("answer")
    if answer not in (None, ""):
        parts.append(f"Answer: {answer}")

    solution = rec.get("solution")
    if solution not in (None, ""):
        parts.append(f"Solution: {solution}")

    return "\n".join(parts)


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


def tokenize_rows(rows, sp):

    eos = sp.eos_id()

    ids = []

    for row in rows:

        text = serialize(row)

        if not text.strip():
            continue

        ids.extend(sp.EncodeAsIds(text))
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

    train_ids = tokenize_rows(train_rows, sp)

    print(f"Train tokens : {len(train_ids):,}")

    print()

    print("Tokenizing validation set...")

    val_ids = tokenize_rows(val_rows, sp)

    print(f"Validation tokens : {len(val_ids):,}")

    print()

    total = len(train_ids) + len(val_ids)

    print(f"TOTAL TOKENS : {total:,}")

    print("=" * 60)

    print("Saving binaries...")

    save_bin(train_ids, TRAIN_BIN)

    save_bin(val_ids, VAL_BIN)

    print("=" * 60)

    print("Finished successfully.")

    print()

    print("Use these with train.py:")

    print(f"--train_bin {TRAIN_BIN}")

    print(f"--val_bin {VAL_BIN}")


if __name__ == "__main__":
    main()