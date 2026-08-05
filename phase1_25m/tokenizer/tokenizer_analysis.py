import os
import io
import json
import re
import math
import random
import unicodedata
from collections import Counter

import sentencepiece as spm

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DATA_PATH = "manifest.repaired.jsonl"  # local copy of the uploaded file
D_MODEL     = 384                       # planned hidden size (embedding-budget math)

SPECIAL_TOKENS = ["<CALC>", "</CALC>"]  # kept atomic; add more markup here if you have it
SPLIT_DIGITS   = True                   # each digit = its own token (good for numerics)

CURRENT_VOCAB   = 8000
CURRENT_CONTEXT = 1024
REC_VOCAB       = 4000
REC_CONTEXT     = 256
VOCAB_SWEEP     = [2000, 3000, 4000, 6000, 8000, 12000,16000, 20000, 30000, 40000, 50000]
CHAR_COVERAGE   = 0.9995
SEED            = 13
WORKDIR         = "tok_analysis_v2"
# ----------------------------------------------------------------------------

random.seed(SEED)
os.makedirs(WORKDIR, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. Load + serialize records the way the model will actually see them
# ----------------------------------------------------------------------------
def serialize(rec):
    """Turn one JSONL record into the flat training string."""
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
        # include any answer keys we didn't explicitly order
        for k, v in ans.items():
            if k not in ANSWER_ORDER and v not in (None, ""):
                parts.append(f"{k}: {v}")
    elif ans not in (None, ""):
        parts.append(f"Answer: {ans}")
    return "\n".join(parts)


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
        f.write(t.replace("\n", " ") + "\n")

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


# ----------------------------------------------------------------------------
# 3. Special-token & digit analysis  (the two data-forced findings)
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("MARKUP & NUMERIC CONTENT")
print("=" * 70)
for tok in SPECIAL_TOKENS:
    cnt = sum(t.count(tok) for t in TEXTS)
    print(f"  {tok!r:>10} occurrences : {cnt:,}  in {sum(1 for t in TEXTS if tok in t):,} samples")
digit_chars = sum(c for ch, c in all_chars.items() if ch.isdigit())
print(f"  digit characters total  : {digit_chars:,} ({100*digit_chars/total_chars:.1f}% of chars)")
print("  -> tags declared as user_defined_symbols stay atomic; digit splitting")
print(f"     is {'ON' if SPLIT_DIGITS else 'OFF'} (recommended ON for numerical answers).")


# ----------------------------------------------------------------------------
# 4. Unicode / symbol inventory + NFKC impact
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
# 4.5 Stray non-target-language character detection
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
print("  -> real unit conflicts = evidence to KEEP lowercasing OFF.")


# ----------------------------------------------------------------------------
# 6. Vocab sweep
# ----------------------------------------------------------------------------
def train_spm(vocab):
    prefix = os.path.join(WORKDIR, f"spm_{vocab}")
    if not os.path.exists(prefix + ".model"):
        spm.SentencePieceTrainer.Train(
            input=corpus_txt,
            model_prefix=prefix,
            model_type="unigram",
            vocab_size=vocab,
            character_coverage=CHAR_COVERAGE,
            byte_fallback=True,
            normalization_rule_name="nfkc",
            add_dummy_prefix=True,
            remove_extra_whitespaces=False,
            split_digits=SPLIT_DIGITS,
            user_defined_symbols=SPECIAL_TOKENS,
            unk_id=0, bos_id=1, eos_id=2, pad_id=3,
        )
    sp = spm.SentencePieceProcessor()
    sp.Load(prefix + ".model")
    return sp

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

sweep = sorted(set(VOCAB_SWEEP) | {CURRENT_VOCAB, REC_VOCAB})
print("\n" + "=" * 70)
print("VOCAB SWEEP  (train 90% / measure held-out 10%)")
print("=" * 70)
print(f"  {'vocab':>6} | {'fertility':>9} | {'chars/tok':>9} | "
      f"{'byte%':>6} | {'used%':>6} | {'emb_params':>10} | {'emb%25M':>7}")
print("  " + "-" * 74)
sweep_results = {}
for v in sweep:
    sp = train_spm(v)
    m = eval_spm(sp, HELD)
    sweep_results[v] = (sp, m)
    emb = v * D_MODEL
    tag = ("  <-current" if v == CURRENT_VOCAB else "") + \
          ("  <-recommended" if v == REC_VOCAB else "")
    print(f"  {v:>6} | {m['fertility']:>9.3f} | {m['chars_per_tok']:>9.2f} | "
          f"{m['byte_fallback_pct']:>6.2f} | {m['vocab_used_pct']:>6.1f} | "
          f"{emb:>10,} | {100*emb/25e6:>6.1f}%{tag}")

# integrity check: are the special tokens actually atomic?
print("\n  special-token integrity (recommended vocab):")
sp_rec = sweep_results[REC_VOCAB][0]
probe = "Substitution: <CALC>1 * 5</CALC>"
print(f"     {probe!r}")
print(f"     -> {sp_rec.EncodeAsPieces(probe)}")


# ----------------------------------------------------------------------------
# 7. Context length
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CONTEXT LENGTH")
print("=" * 70)
tl = eval_spm(sp_rec, TEXTS)["tok_lens"]
for p in (50, 90, 95, 99, 99.9, 100):
    print(f"  token-length p{p:<5}: {pct(tl, p)}")
p999 = pct(tl, 99.9)
def next_pow2(x): return 1 << (max(1, x - 1)).bit_length()
data_ctx = next_pow2(p999)
print(f"\n  longest = {max(tl)} tokens; p99.9 = {p999}; data needs ~{data_ctx}.")
print(f"  current={CURRENT_CONTEXT}  recommended={REC_CONTEXT}")
if data_ctx <= REC_CONTEXT:
    print(f"  -> {REC_CONTEXT} covers data with margin; 1024 wastes "
          f"~{CURRENT_CONTEXT//REC_CONTEXT}x attention compute for zero gain.")
else:
    print(f"  -> data needs {data_ctx}; set context to at least that.")


# ----------------------------------------------------------------------------
# 8. Verdict
# ----------------------------------------------------------------------------
cur_m, rec_m = sweep_results[CURRENT_VOCAB][1], sweep_results[REC_VOCAB][1]
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
def verdict(name, change, msg):
    print(f"  [{'CHANGE' if change else 'OK   '}] {name}: {msg}")

verdict("Vocab 8000 -> 4000", rec_m["vocab_used_pct"] > cur_m["vocab_used_pct"] + 5,
        f"8k used={cur_m['vocab_used_pct']:.0f}%/fert={cur_m['fertility']:.2f}  "
        f"4k used={rec_m['vocab_used_pct']:.0f}%/fert={rec_m['fertility']:.2f}. "
        f"Frees {(CURRENT_VOCAB-REC_VOCAB)*D_MODEL:,} params.")
verdict("Context 1024 -> 256", data_ctx <= REC_CONTEXT,
        f"data needs ~{data_ctx}; smaller context = cheaper attention + bigger batch.")
verdict("Add <CALC> special tokens", True, "forced by your markup; keeps tags atomic.")
verdict("split_digits", SPLIT_DIGITS, "recommended ON for numerical answers.")
verdict("Byte fallback ON", False,
        f"byte tokens={cur_m['byte_fallback_pct']:.1f}% -> keep ON.")
verdict("Lowercasing OFF", False,
        f"{len(unit_conflicts)} unit case conflicts -> keep OFF.")
verdict("NFKC", changed_samples/n > 0.25,
        f"{100*changed_samples/n:.1f}% samples change; confirm symbols survive.")

print("\nArtifacts in ./%s/ (spm_*.model reusable for training)." % WORKDIR)
print("Done.")