import os
import io
import json
import re
import math
import random
import unicodedata
import hashlib
from collections import Counter

import sentencepiece as spm
import numpy as np

# ----------------------------------------------------------------------------
# CONFIG (architecture locked)
# ----------------------------------------------------------------------------
DATA_PATH = "manifest.repaired.jsonl"  # local copy of the uploaded file
D_MODEL = 1536                  # updated: hidden_size=1536 (10-layer arch)
CURRENT_VOCAB   = 11000                 # locked spec: vocab_size=11000 (tokenizer finalized)
CURRENT_CONTEXT = 2048                  # locked spec: max_position_embeddings=2048

SPLIT_DIGITS   = True                   # each digit = its own token (good for numerics)
CHAR_COVERAGE  = 1.0                    # locked tokenizer config
SEED           = 42                     # single global seed: reuse across tokenizer
                                          # training, dataset prep, and model training
WORKDIR        = "tok_analysis_v2"
MAX_SAMPLE_LEN = 4000                   # chars; flag samples longer than this
# ----------------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
# NOTE: when you get to the training script, also set:
#   torch.manual_seed(SEED)
#   torch.cuda.manual_seed_all(SEED)
# using this same SEED, so shuffling/tokenizer-prep/training all align.
os.makedirs(WORKDIR, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. Load + serialize records the way the model will actually see them
# ----------------------------------------------------------------------------
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
    """
    s = _invisible_re.sub("", s)
    s = _whitespace_re.sub(" ", s)
    return s


def serialize(rec):
    """
    Converts one dataset record into plain text for tokenizer training.
    The tokenizer only sees this text.

    Handles both record types in manifest.repaired.jsonl:
      - objective (MCQ): has a non-empty "options" dict
      - subjective: "options" absent/empty, falls through cleanly since the
        Options: section is only emitted when options is a non-empty dict.

    Answer and Solution are merged into a single "Answer:" section so the
    model does not treat them as two distinct identities -- they represent
    one unified "correct answer + reasoning" concept in this dataset. The
    answer (e.g. an MCQ letter) sits on its own line right after the
    "Answer:" marker, and the explanation (when present) follows as a
    blank-line-separated paragraph underneath. This keeps the model always
    predicting immediately after "Answer:", lets explanations naturally
    follow, and avoids a separate "Solution:" field that many samples lack.

    Fields ignored (id, subject, route, source, repaired, has_solution) don't
    contribute to language-modeling text and are left untouched elsewhere in
    the script (e.g. "id" is still used for stray-character tracing below).
    """
    parts = []
    # ------------------------
    # Question
    # ------------------------
    question = rec.get("question")
    if question:
        parts.append(f"Question:\n{question}")
    # ------------------------
    # Options (MCQ)
    # ------------------------
    options = rec.get("options")
    if isinstance(options, dict) and options:
        opt_lines = ["Options:"]
        for key in sorted(options.keys()):
            value = options[key]
            if value not in (None, ""):
                opt_lines.append(f"{key}) {value}")
        parts.append("\n".join(opt_lines))
    # ------------------------
    # Answer (+ explanation, if available) as one unified section
    # ------------------------
    answer = rec.get("answer")
    solution = rec.get("solution")
    if answer not in (None, ""):
        answer_block = f"Answer:\n{answer}"
        if solution not in (None, ""):
            answer_block += f"\n\n{solution}"
        parts.append(answer_block)
    elif solution not in (None, ""):
        # rare case: explanation present with no explicit answer field
        parts.append(f"Answer:\n{solution}")
    return clean_text("\n\n".join(parts))


def load_texts():
    texts = []
    raw_records = []
    with io.open(DATA_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            s = serialize(rec)
            if s.strip():
                texts.append(s)
                raw_records.append(rec)
    if not texts:
        raise SystemExit("Loaded 0 texts. Check DATA_PATH / record shape.")
    return texts, raw_records


print("Loading corpus...")
TEXTS, RAW_RECORDS = load_texts()
# shuffle texts and records together so we can still trace a flagged sample
# back to its original record id for the stray-character report below
paired = list(zip(TEXTS, RAW_RECORDS))
random.shuffle(paired)
TEXTS, RAW_RECORDS = [p[0] for p in paired], [p[1] for p in paired]
n = len(TEXTS)
split = int(n * 0.9)
TRAIN, HELD = TEXTS[:split], TEXTS[split:]

corpus_txt = os.path.join(WORKDIR, "corpus.txt")
with io.open(corpus_txt, "w", encoding="utf-8") as f:
    for t in TRAIN:
        # keep real newlines -- SentencePiece learns punctuation/whitespace
        # statistics, and collapsing "Question:\n...\nOptions:\n..." into one
        # line destroys structure the tokenizer could otherwise pick up on.
        f.write(t)
        f.write("\n\n")

print(f"  loaded {n:,} serialized samples")
print("  --- example serialized sample ---")
print("  " + TEXTS[0].replace("\n", "\n  ")[:600])
print("  ---------------------------------")


# ----------------------------------------------------------------------------
# 2. Corpus statistics
# ----------------------------------------------------------------------------
def word_tokens(s):
    return re.findall(r"\w+|[^\w\s]", s, flags=re.UNICODE)

all_chars   = Counter()
word_counts = Counter()
word_lens, char_lens = [], []
for t in TEXTS:
    all_chars.update(t)
    ws = word_tokens(t)
    word_counts.update(w.lower() for w in ws)
    word_lens.append(len(ws))
    char_lens.append(len(t))

total_words = sum(word_lens)
total_chars = sum(char_lens)

def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0
    k = min(len(xs) - 1, int(math.ceil(p / 100.0 * len(xs))) - 1)
    return xs[max(0, k)]

print("\n" + "=" * 70)
print("CORPUS OVERVIEW")
print("=" * 70)
print(f"  samples                  : {n:,}")
print(f"  total words              : {total_words:,}")
print(f"  total chars              : {total_chars:,}")
print(f"  unique lowercased words  : {len(word_counts):,}")
print(f"  words/sample mean/p50/p95/p99/max : "
      f"{total_words/n:.1f} / {pct(word_lens,50)} / {pct(word_lens,95)} / "
      f"{pct(word_lens,99)} / {max(word_lens)}")

cum, cov95, cov99 = 0, None, None
for i, (_, c) in enumerate(word_counts.most_common(), 1):
    cum += c
    if cov95 is None and cum >= 0.95 * total_words: cov95 = i
    if cov99 is None and cum >= 0.99 * total_words: cov99 = i
print(f"  word-types to cover 95% / 99% of tokens : {cov95:,} / {cov99:,}")

# objective (MCQ) vs subjective split, and average answer/solution length
n_objective = sum(1 for rec in RAW_RECORDS
                   if isinstance(rec.get("options"), dict) and rec.get("options"))
n_subjective = n - n_objective
answer_lens = [len(str(rec["answer"])) for rec in RAW_RECORDS
               if rec.get("answer") not in (None, "")]
solution_lens = [len(str(rec["solution"])) for rec in RAW_RECORDS
                  if rec.get("solution") not in (None, "")]
print(f"  objective (MCQ) samples  : {n_objective:,} ({100*n_objective/n:.1f}%)")
print(f"  subjective samples       : {n_subjective:,} ({100*n_subjective/n:.1f}%)")
avg_answer_len = sum(answer_lens) / len(answer_lens) if answer_lens else 0
avg_solution_len = sum(solution_lens) / len(solution_lens) if solution_lens else 0
print(f"  avg answer length (chars)   : {avg_answer_len:.1f}" if answer_lens else
      "  avg answer length (chars)   : n/a")
print(f"  avg solution length (chars) : {avg_solution_len:.1f}" if solution_lens else
      "  avg solution length (chars) : n/a")


# ----------------------------------------------------------------------------
# 3. Unicode / symbol inventory + NFKC impact
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("UNICODE & SYMBOL INVENTORY  (°, µ, Ω, ×, ², units...)")
print("=" * 70)
non_ascii = Counter({ch: c for ch, c in all_chars.items() if ord(ch) > 127})
print(f"  distinct non-ASCII chars : {len(non_ascii)}")
for ch, c in non_ascii.most_common(25):
    print(f"     {ch!r:>6}  U+{ord(ch):04X}  {unicodedata.name(ch,'?')[:30]:<30} {c:,}")
changed_samples = sum(1 for t in TEXTS if unicodedata.normalize("NFKC", t) != t)
print(f"  samples altered by NFKC  : {changed_samples:,} ({100*changed_samples/n:.1f}%)")
print("  -> confirm × (U+00D7) and any exponents survive NFKC as you expect.")

# concrete before/after spot-check on real exponent-bearing samples, so you can
# actually SEE what NFKC does to your physics content instead of trusting a
# percentage blindly. cm² / m³ style superscripts are exactly the case NFKC is
# known to alter (folds superscript digits to plain digits), which may or may
# not be what you want depending on how your model is meant to represent units.
print("\n  NFKC before/after spot-check on exponent-bearing samples:")
exponent_examples = [t for t in TEXTS if re.search(r"[²³]", t)][:5]
if exponent_examples:
    for ex in exponent_examples:
        before = ex[:90].replace("\n", " ")
        after = unicodedata.normalize("NFKC", ex)[:90].replace("\n", " ")
        marker = " <-- CHANGED" if before != after else ""
        print(f"     before: {before!r}")
        print(f"     after : {after!r}{marker}")
else:
    print("     (no exponent-bearing samples found to spot-check)")


# ----------------------------------------------------------------------------
# 4. Stray non-target-language character detection
# ----------------------------------------------------------------------------
# Beyond the expected physics symbol set (Greek letters, degree/multiplication/
# division signs, superscripts), any Cyrillic or CJK characters showing up in
# an English/Hindi-curriculum physics dataset are almost certainly generation
# noise (rare tokenizer artifacts leaking through from the LLM that generated
# this data), not real intended content. Report exactly which samples contain
# them so you can decide to fix or drop those specific rows before training --
# a handful of stray characters can otherwise silently bloat your vocabulary
# with junk tokens that will never generalize.
print("\n" + "=" * 70)
print("STRAY NON-TARGET-LANGUAGE CHARACTERS (likely generation noise)")
print("=" * 70)

def is_cyrillic(ch):
    return "\u0400" <= ch <= "\u04FF"

def is_cjk(ch):
    return "\u4E00" <= ch <= "\u9FFF"

flagged_samples = []
for t, rec in zip(TEXTS, RAW_RECORDS):
    stray_chars = sorted({ch for ch in t if is_cyrillic(ch) or is_cjk(ch)})
    if stray_chars:
        flagged_samples.append((rec, stray_chars, t))

print(f"  samples containing stray Cyrillic/CJK characters : {len(flagged_samples)}")
if flagged_samples:
    print("  flagged sample ids and characters found:")
    for rec, stray_chars, t in flagged_samples[:20]:
        rec_id = rec.get("id") or rec.get("source_concept_id") or "?"
        snippet = t[:80].replace("\n", " ")
        print(f"     id={rec_id!r}  chars={stray_chars}  text={snippet!r}...")
    if len(flagged_samples) > 20:
        print(f"     ... and {len(flagged_samples) - 20} more (see flagged_ids.txt)")
    with io.open(os.path.join(WORKDIR, "flagged_ids.txt"), "w", encoding="utf-8") as f:
        for rec, stray_chars, t in flagged_samples:
            rec_id = rec.get("id") or rec.get("source_concept_id") or "?"
            f.write(f"{rec_id}\t{stray_chars}\n")
    print(f"  -> Full list of flagged ids written to {WORKDIR}/flagged_ids.txt")
else:
    print("  -> none found, clean.")


# ----------------------------------------------------------------------------
# 5. Case-sensitivity evidence
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CASE SENSITIVITY  (units where case matters: A, V, K, N, W, J ...)")
print("=" * 70)
surface = Counter()
for t in TEXTS:
    surface.update(word_tokens(t))
by_lower = {}
for w in surface:
    by_lower.setdefault(w.lower(), set()).add(w)
UNIT_LIKE = {"a","v","k","n","w","j","c","t","g","m","s","pa","hz","kg","mv"}
unit_conflicts = [(lw, forms) for lw, forms in by_lower.items()
                  if len(forms) > 1 and lw in UNIT_LIKE]
print(f"  word-forms differing only by case : "
      f"{sum(1 for f in by_lower.values() if len(f) > 1):,}")
for lw, forms in sorted(unit_conflicts)[:15]:
    print(f"     {lw!r:>6} -> {sorted(forms)}")
print("  -> real unit conflicts confirm lowercasing stays OFF.")


# ----------------------------------------------------------------------------
# 6. Pre-training corpus validation
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CORPUS VALIDATION")
print("=" * 70)

# 6a. Duplicate samples (exact text match, via hash to keep memory sane)
hash_counts = Counter(hashlib.sha256(t.encode("utf-8")).hexdigest() for t in TEXTS)
dup_hashes = {h: c for h, c in hash_counts.items() if c > 1}
n_dup_samples = sum(c for c in dup_hashes.values())
n_dup_groups = len(dup_hashes)
print(f"  duplicate samples        : {n_dup_samples:,} ({100*n_dup_samples/n:.2f}%) "
      f"across {n_dup_groups:,} duplicate groups")

# 6b. Empty question/answer (post-serialization, catches missing required fields)
n_missing_question = sum(1 for rec in RAW_RECORDS if not rec.get("question"))
n_missing_answer = sum(
    1 for rec in RAW_RECORDS
    if rec.get("answer") in (None, "") and rec.get("solution") in (None, "")
)
print(f"  samples missing question : {n_missing_question:,}")
print(f"  samples missing answer+solution : {n_missing_answer:,}")

# 6c. UTF-8 validation (re-encode/decode round-trip check on raw file bytes)
utf8_errors = 0
with io.open(DATA_PATH, "rb") as f:
    for raw_line in f:
        try:
            raw_line.decode("utf-8")
        except UnicodeDecodeError:
            utf8_errors += 1
print(f"  lines with UTF-8 decode errors : {utf8_errors:,}")

# 6d. Maximum sample length (char-based; token-based check follows after
# the tokenizer is trained below, since char count alone can be misleading --
# a 2500-char sample may still fit in 2048 tokens while a shorter one may not)
over_max_idx = [i for i, ln in enumerate(char_lens) if ln > MAX_SAMPLE_LEN]
print(f"  samples over {MAX_SAMPLE_LEN} chars : {len(over_max_idx):,} "
      f"(longest = {max(char_lens):,} chars)")

# 6e. ASCII control character detection (excluding \n and \t), common in
# scraped/OCR'd datasets and otherwise invisible sources of vocabulary noise
def find_control_chars(s):
    return sorted({ch for ch in s if ord(ch) < 32 and ch not in ("\n", "\t")})

control_flagged = []
for t, rec in zip(TEXTS, RAW_RECORDS):
    ctrl = find_control_chars(t)
    if ctrl:
        control_flagged.append((rec, ctrl))
print(f"  samples with control characters : {len(control_flagged):,}")
if control_flagged:
    for rec, ctrl in control_flagged[:10]:
        rec_id = rec.get("id") or rec.get("source_concept_id") or "?"
        codepoints = [f"U+{ord(c):04X}" for c in ctrl]
        print(f"     id={rec_id!r}  control_chars={codepoints}")
    if len(control_flagged) > 10:
        print(f"     ... and {len(control_flagged) - 10} more")

if n_dup_samples or n_missing_question or n_missing_answer or utf8_errors or control_flagged:
    print("  -> issues found above should be resolved or explicitly accepted")
    print("     before training on this corpus.")
else:
    print("  -> corpus passes all validation checks.")


# ----------------------------------------------------------------------------
# 7. Train tokenizer (vocab and context are locked, no sweep/selection)
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"TRAINING TOKENIZER  (vocab={CURRENT_VOCAB})")
print("=" * 70)

model_prefix = os.path.join(WORKDIR, f"spm_{CURRENT_VOCAB}")
if not os.path.exists(model_prefix + ".model"):
    spm.SentencePieceTrainer.Train(
        input=corpus_txt,
        model_prefix=model_prefix,
        model_type="unigram",
        vocab_size=CURRENT_VOCAB,
        character_coverage=CHAR_COVERAGE,
        byte_fallback=True,
        normalization_rule_name="nfkc",
        add_dummy_prefix=True,
        remove_extra_whitespaces=False,
        split_digits=SPLIT_DIGITS,
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
    )
sp = spm.SentencePieceProcessor()
sp.Load(model_prefix + ".model")
print(f"  -> trained model saved to {model_prefix}.model")


def eval_spm(sp, texts):
    n_tok = n_word = n_char = byte_toks = 0
    used = set()
    tok_lens = []
    for t in texts:
        ids = sp.EncodeAsIds(t)
        pieces = sp.EncodeAsPieces(t)
        n_tok += len(ids); tok_lens.append(len(ids))
        n_word += len(word_tokens(t)); n_char += len(t)
        used.update(ids)
        byte_toks += sum(1 for p in pieces if len(p) == 6 and p.startswith("<0x"))
    return {
        "fertility": n_tok / max(1, n_word),
        "chars_per_tok": n_char / max(1, n_tok),
        "byte_fallback_pct": 100 * byte_toks / max(1, n_tok),
        "vocab_used_pct": 100 * len(used) / sp.GetPieceSize(),
        "tok_lens": tok_lens,
    }

held_metrics = eval_spm(sp, HELD)
print(f"  held-out fertility       : {held_metrics['fertility']:.3f}")
print(f"  held-out chars/token     : {held_metrics['chars_per_tok']:.2f}")
print(f"  held-out byte-fallback % : {held_metrics['byte_fallback_pct']:.2f}")
print(f"  held-out vocab used %    : {held_metrics['vocab_used_pct']:.1f}")
print(f"  embedding params (vocab x d_model) : {CURRENT_VOCAB * D_MODEL:,}")

# fixed context length -- report how the corpus fits, but no auto-recommendation
full_metrics = eval_spm(sp, TEXTS)
tl = full_metrics["tok_lens"]
n_over_context = sum(1 for l in tl if l > CURRENT_CONTEXT)
print(f"\n  fixed context length      : {CURRENT_CONTEXT}")
print(f"  samples exceeding context : {n_over_context:,} / {n:,} "
      f"({100*n_over_context/n:.2f}%)")
print(f"  token-length p50/p95/p99/max : "
      f"{pct(tl,50)} / {pct(tl,95)} / {pct(tl,99)} / {max(tl)}")

# token-based view of the char-flagged long samples from section 6d --
# the model trains on tokens, not characters, so a sample flagged for char
# length may still fit comfortably in context (or vice versa)
if over_max_idx:
    over_max_tok_lens = [tl[i] for i in over_max_idx]
    n_also_over_context = sum(1 for l in over_max_tok_lens if l > CURRENT_CONTEXT)
    print(f"\n  of the {len(over_max_idx):,} samples over {MAX_SAMPLE_LEN} chars:")
    print(f"     token-length min/mean/max : "
          f"{min(over_max_tok_lens)} / {sum(over_max_tok_lens)/len(over_max_tok_lens):.1f} / "
          f"{max(over_max_tok_lens)}")
    print(f"     of those, {n_also_over_context:,} also exceed the "
          f"{CURRENT_CONTEXT}-token context window")


# ----------------------------------------------------------------------------
# 8. Tokenizer configuration dump (for reproducibility)
# ----------------------------------------------------------------------------
tokenizer_config = {
    "d_model": D_MODEL,
    "vocab_size": CURRENT_VOCAB,
    "max_position_embeddings": CURRENT_CONTEXT,
    "model_type": "unigram",
    "character_coverage": CHAR_COVERAGE,
    "byte_fallback": True,
    "normalization_rule_name": "nfkc",
    "add_dummy_prefix": True,
    "remove_extra_whitespaces": False,
    "split_digits": SPLIT_DIGITS,
    "lowercase": False,
    "special_ids": {"unk_id": 0, "bos_id": 1, "eos_id": 2, "pad_id": 3},
    "seed": SEED,
    "model_path": model_prefix + ".model",
    "held_out_metrics": {
        "fertility": held_metrics["fertility"],
        "chars_per_tok": held_metrics["chars_per_tok"],
        "byte_fallback_pct": held_metrics["byte_fallback_pct"],
        "vocab_used_pct": held_metrics["vocab_used_pct"],
    },
}
config_path = os.path.join(WORKDIR, "tokenizer_config.json")
with io.open(config_path, "w", encoding="utf-8") as f:
    json.dump(tokenizer_config, f, indent=2)
print(f"\n  -> tokenizer config written to {config_path}")

print("\nArtifacts in ./%s/ (spm_%s.model + tokenizer_config.json)." % (WORKDIR, CURRENT_VOCAB))
print("Done.")