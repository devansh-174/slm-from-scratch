"""
Tokenizes train.jsonl (and val.jsonl, if present) into the .bin token-id
files train.py expects (np.uint16 arrays -- safe since vocab_size=12736
fits comfortably under 65536).

If val.jsonl does NOT exist, this auto-splits train.jsonl by
source_concept_id (falls back to random row split if that field is
absent) so paraphrases of the same concept never leak across train/val.

Uses the SAME serialize() logic as train_tokenizer.py so the token
stream the model trains on matches exactly what the tokenizer was
fit on.

USAGE:
    python3 make_bins.py
"""

import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import sentencepiece as spm

TRAIN_JSONL = Path("train.jsonl")
VAL_JSONL = Path("val.jsonl")
TOKENIZER_MODEL = Path("tokenizer/slm_tokenizer_12779.model")

TRAIN_BIN = Path("train.bin")
VAL_BIN = Path("val.bin")

VAL_FRACTION = 0.10
SEED = 42

ANSWER_ORDER = ["given", "formula", "substitution", "explanation", "final_answer"]


def serialize(rec):
    """Identical to train_tokenizer.py's serialize() -- must match exactly
    so token statistics/behavior stay consistent with what the tokenizer saw."""
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


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_train_val(rows, val_fraction, seed):
    """Split by source_concept_id if present (avoids paraphrase leakage);
    falls back to a plain random row split if that field is missing."""
    if rows and "source_concept_id" in rows[0]:
        by_concept = defaultdict(list)
        for r in rows:
            by_concept[r["source_concept_id"]].append(r)
        concept_ids = list(by_concept.keys())
        random.Random(seed).shuffle(concept_ids)
        val_count = max(1, int(len(concept_ids) * val_fraction))
        val_ids = set(concept_ids[:val_count])
        train_rows, val_rows = [], []
        for cid, group in by_concept.items():
            (val_rows if cid in val_ids else train_rows).extend(group)
        print(f"Split by source_concept_id: {len(concept_ids) - len(val_ids)} train concepts, "
              f"{len(val_ids)} val concepts")
    else:
        rows = rows[:]
        random.Random(seed).shuffle(rows)
        val_count = max(1, int(len(rows) * val_fraction))
        val_rows = rows[:val_count]
        train_rows = rows[val_count:]
        print("No source_concept_id field found -- used a plain random row split instead "
              "(WARNING: if this data has paraphrase groups, this risks train/val leakage)")

    return train_rows, val_rows


def tokenize_rows(rows, sp):
    """Serializes + tokenizes every row, inserting EOS between examples,
    and returns one flat list of token ids."""
    all_ids = []
    eos_id = sp.eos_id()
    for row in rows:
        text = serialize(row)
        if not text.strip():
            continue
        ids = sp.EncodeAsIds(text)
        all_ids.extend(ids)
        all_ids.append(eos_id)
    return all_ids


def save_bin(ids, path):
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(str(path))
    print(f"  wrote {path}: {len(arr):,} tokens ({path.stat().st_size / 1e6:.1f} MB)")


def main():
    print(f"Loading tokenizer from {TOKENIZER_MODEL} ...")
    sp = spm.SentencePieceProcessor()
    sp.Load(str(TOKENIZER_MODEL))
    print(f"  vocab_size={sp.GetPieceSize()}")

    print(f"\nLoading {TRAIN_JSONL} ...")
    train_rows = load_jsonl(TRAIN_JSONL)
    print(f"  {len(train_rows):,} rows")

    if VAL_JSONL.exists():
        print(f"\n{VAL_JSONL} found -- using it directly (no auto-split).")
        val_rows = load_jsonl(VAL_JSONL)
        print(f"  {len(val_rows):,} rows")
    else:
        print(f"\n{VAL_JSONL} NOT found -- auto-splitting {TRAIN_JSONL} "
              f"({VAL_FRACTION:.0%} held out for validation) ...")
        train_rows, val_rows = split_train_val(train_rows, VAL_FRACTION, SEED)
        print(f"  train: {len(train_rows):,} rows | val: {len(val_rows):,} rows")

    print(f"\nTokenizing train split ...")
    train_ids = tokenize_rows(train_rows, sp)
    print(f"  {len(train_ids):,} total tokens")

    print(f"\nTokenizing val split ...")
    val_ids = tokenize_rows(val_rows, sp)
    print(f"  {len(val_ids):,} total tokens")

    print(f"\nSaving .bin files ...")
    save_bin(train_ids, TRAIN_BIN)
    save_bin(val_ids, VAL_BIN)

    print(f"\nDone. Ready to run train.py with:")
    print(f"  --train_bin {TRAIN_BIN}")
    print(f"  --val_bin {VAL_BIN}")


if __name__ == "__main__":
    main()