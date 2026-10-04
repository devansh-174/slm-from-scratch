"""
Independent verification of the tokenizer produced by
train_tokenizer2.py.

Unlike train_tokenizer2.py, this script does NOT trust any statistics
printed during training. It loads the generated SentencePiece model
directly from disk and verifies every important property from scratch.

Checks performed include:

1. Vocabulary size
2. Special token IDs
3. Custom tokens / vocabulary statistics
4. Placeholder / padding pieces
5. Encode-decode round-trip integrity
6. UNK usage
7. Digit splitting + numeric / scientific-notation round-trip
8. Physics & Mathematics symbols
9. Unicode (NFKC) normalization
10. Byte fallback
11. Serialization headers
12. Case sensitivity
13. Physics units
14. Vocabulary cleanliness
15. Performance benchmark (informational)
16. Vocabulary utilization (optional held-out corpus, informational)
17. Context statistics (optional held-out corpus, informational)
18. Fertility (optional held-out corpus)
19. Characters per token (optional held-out corpus)

USAGE:
    python3 verify_tokenizer.py tokenizer/model300m.model
    python3 verify_tokenizer.py tokenizer/model300m.model tokenizer/train_corpus.txt
        (optional second arg: a held-out text file, one sample per line or
        blank-line separated, used for vocabulary utilization / context /
        fertility / chars-per-token stats -- skipped if not provided)
"""

import sys
import re
import io
import os
import json
import time
import math
import hashlib
import unicodedata
from datetime import datetime

import sentencepiece as spm

try:
    from sentencepiece import sentencepiece_model_pb2 as sp_model_pb2
except ImportError:
    sp_model_pb2 = None

# ---------------------------------------------------------------------------
# Configuration / expectations
# ---------------------------------------------------------------------------

EXPECTED_CUSTOM_TOKENS = []
EXPECTED_VOCAB = 11000
EXPECTED_CONTEXT = 2048

# If populated, special token IDs must match EXACTLY (strict check for a
# locked tokenizer spec). Leave as {} to fall back to the looser
# "just needs to be present (>= 0) and unk is unique" check.
EXPECTED_SPECIAL_IDS = {
    "unk": 0,
    "bos": 1,
    "eos": 2,
    "pad": 3,
}

# Result statuses. Not everything is a pass/fail check -- some checks are
# purely informational (benchmarks, corpus stats) and shouldn't be able to
# flip the overall verdict to FAIL just because a number came back low/high.
PASS = "PASS"
FAIL = "FAIL"
INFO = "INFO"
SKIPPED = "SKIPPED"


# ---------------------------------------------------------------------------
# Probe corpora, grouped and named so they're easy to extend independently
# ---------------------------------------------------------------------------

PROBE_SERIALIZED_QA = [
    """Question: The magnetic force on a moving charged particle can change the particle's

Options:
A) speed
B) direction
C) Both of these
D) Neither of these

Answer:
B""",
    """Question: A car travels 60 km in 2 hours. What is its average speed?

Answer:
30 km/h

Speed = Distance / Time = 30 km/h""",
    """Question: Why does ice float on water?

Answer:
Ice has lower density than liquid water.""",
    """Question: What is the SI unit of force?

Answer:
Newton""",
]

# Real textbook constants. Superscript scientific notation is intentionally
# expressed with caret notation here since superscript digits (⁵, ⁻, ⁸, ...)
# are NFKC-normalized to their plain-digit/ASCII equivalents by the
# tokenizer's normalizer and therefore do not round-trip byte-for-byte.
PROBE_NUMERICS = [
    "9.81",
    "9.81 m/s2",
    "10^5",
    "10^-5",
    "3.14",
    "6×10^8",
    "3×10^8 m/s",
    "6.67×10^-11 N·m2/kg2",
    "1.602×10^-19 C",
    "2.998×10^8",
]

PROBE_SYMBOLS = [
    # Greek letters
    "Ω", "μ", "α", "β", "γ", "Δ", "λ", "π", "θ", "φ",
    # Operators / relations
    "°", "±", "×", "÷", "≤", "≥", "≈", "≠", "≡", "∝",
    # Roots / infinity
    # (² and ³ are intentionally excluded: under NFKC normalization they are
    # normalized to "2" and "3" respectively, so they are not expected to
    # round-trip to their original superscript form.)
    "√", "∞",
    # Calculus / vector calculus
    "∑", "∫", "∂", "∇",
    # Set theory / logic
    "∈", "∉", "⊂", "⊆", "∀", "∃",
    # Arrows
    "→", "←", "↔",
]

PROBE_UNITS = [
    "kg", "m/s", "m²", "cm²", "kg·m/s²", "N", "J", "W", "Hz", "Pa", "eV",
    "kg/m³", "mol", "A", "V", "C", "F", "H", "T", "lx", "Ω",
]

PROBE_HEADERS = ["Question:", "Options:", "Answer:"]
PROBE_CASE_PAIRS = [("V", "v"), ("A", "a"), ("K", "k")]
PROBE_NFKC = [
    ("Ａ", "A"),
    ("㎏", "kg"),
    ("１", "1"),
]
PROBE_RARE = "🧪🛰️"


def word_tokens(s):
    return re.findall(r"\w+|[^\w\s]", s, flags=re.UNICODE)


def nfkc(s):
    return unicodedata.normalize("NFKC", s)


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0
    k = min(len(xs) - 1, int(math.ceil(p / 100.0 * len(xs))) - 1)
    return xs[max(0, k)]


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 verify_tokenizer.py <model_path> [held_out_corpus.txt]")
        sys.exit(1)

    model_path = sys.argv[1]
    corpus_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(model_path):
        print(f"Model file does not exist: {model_path}")
        print("Check the path and try again -- verification cannot proceed without a tokenizer model file.")
        sys.exit(1)

    print(f"Loading {model_path} directly with SentencePieceProcessor ...")
    sp = spm.SentencePieceProcessor()
    try:
        if not sp.Load(model_path):
            print(f"Failed to load tokenizer: {model_path}")
            sys.exit(1)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        sys.exit(1)

    print(f"SentencePiece library version: {spm.__version__}")

    failures = []
    report = {"model_path": model_path, "sentencepiece_version": spm.__version__}
    report["verified_at"] = datetime.now().isoformat()
    report["expected"] = {
        "vocab_size": EXPECTED_VOCAB,
        "context": EXPECTED_CONTEXT,
        "special_ids": EXPECTED_SPECIAL_IDS or None,
    }
    with open(model_path, "rb") as f:
        model_bytes = f.read()
    report["model_sha256"] = hashlib.sha256(model_bytes).hexdigest()

    # ---- Tokenizer configuration (for reproducibility) ----
    tokenizer_config = None
    if sp_model_pb2 is not None:
        try:
            model_proto = sp_model_pb2.ModelProto()
            model_proto.ParseFromString(sp.serialized_model_proto())
            tokenizer_config = {
                "vocab_size": model_proto.trainer_spec.vocab_size,
                "normalization": model_proto.normalizer_spec.name,
                "byte_fallback": model_proto.trainer_spec.byte_fallback,
                "split_digits": model_proto.trainer_spec.split_digits,
                "character_coverage": model_proto.trainer_spec.character_coverage,
            }
        except Exception as e:
            tokenizer_config = {"error": f"Could not parse model proto: {e}"}
    else:
        tokenizer_config = {"error": "sentencepiece_model_pb2 not importable"}
    report["tokenizer_config"] = tokenizer_config

    categories = {}  # category_name -> PASS / FAIL / INFO / SKIPPED

    # ---- Check 1: report the actual piece count and verify against expected ----
    actual_size = sp.GetPieceSize()
    print(f"\n[1] GetPieceSize() = {actual_size}  (this is the real, final vocab size)")
    if actual_size != EXPECTED_VOCAB:
        failures.append(f"Vocabulary mismatch. Expected {EXPECTED_VOCAB}, got {actual_size}")
        categories["Vocabulary Size"] = FAIL
    else:
        print(f"    OK -- vocabulary size = {EXPECTED_VOCAB}")
        categories["Vocabulary Size"] = PASS
    report["vocab_size"] = actual_size

    # ---- Check 2: special token IDs are exactly where expected ----
    print(f"\n[2] Special token IDs (unk/bos/eos/pad):")
    special_ids = {
        "unk": sp.unk_id(),
        "bos": sp.bos_id(),
        "eos": sp.eos_id(),
        "pad": sp.pad_id(),
    }
    for name, tid in special_ids.items():
        print(f"    {name}_id() = {tid}")

    special_ok = True
    if all(tid >= 0 for tid in special_ids.values()):
        print("    OK -- unk/bos/eos/pad are all present and valid")
    else:
        for name, tid in special_ids.items():
            if tid < 0:
                failures.append(f"Missing {name.upper()} token")
        special_ok = False

    if list(special_ids.values()).count(sp.unk_id()) == 1:
        print("    OK -- exactly ONE dedicated <unk> slot, no extra unknown tokens")
    else:
        failures.append("More than one special token maps to the <unk> id")
        special_ok = False

    if EXPECTED_SPECIAL_IDS:
        mismatches = {
            name: (special_ids[name], expected)
            for name, expected in EXPECTED_SPECIAL_IDS.items()
            if special_ids.get(name) != expected
        }
        if mismatches:
            for name, (got, expected) in mismatches.items():
                failures.append(
                    f"Special token id mismatch for {name}: expected {expected}, got {got}"
                )
            print(f"    FAIL -- strict special-id spec violated: {mismatches}")
            special_ok = False
        else:
            print(f"    OK -- special IDs match locked spec {EXPECTED_SPECIAL_IDS}")

    report["special_ids"] = special_ids
    report["special_pieces"] = {
        "unk": sp.IdToPiece(sp.unk_id()),
        "bos": sp.IdToPiece(sp.bos_id()),
        "eos": sp.IdToPiece(sp.eos_id()),
        "pad": sp.IdToPiece(sp.pad_id()),
    }
    categories["Special Tokens"] = PASS if special_ok else FAIL

    # ---- Check 3: custom tokens + vocabulary statistics ----
    print(f"\n[3] Custom tokens:")
    all_pieces = [sp.IdToPiece(i) for i in range(actual_size)]

    byte_piece_re = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
    normal_pieces = [p for p in all_pieces if not byte_piece_re.match(p)]
    byte_pieces = [p for p in all_pieces if byte_piece_re.match(p)]
    vocab_stats = {
        "average_piece_length": sum(len(p) for p in normal_pieces) / max(1, len(normal_pieces)),
        "longest_piece_length": max(len(p) for p in normal_pieces),
        "shortest_piece_length": min(len(p) for p in normal_pieces),
        "byte_piece_count": len(byte_pieces),
        "normal_piece_count": len(normal_pieces),
    }
    report["vocabulary_statistics"] = vocab_stats
    print(f"    average piece length : {vocab_stats['average_piece_length']:.2f}")
    print(f"    longest piece length  : {vocab_stats['longest_piece_length']}")
    print(f"    shortest piece length : {vocab_stats['shortest_piece_length']}")
    print(f"    byte pieces           : {vocab_stats['byte_piece_count']}")
    print(f"    normal pieces         : {vocab_stats['normal_piece_count']}")

    # duplicate vocabulary piece detection
    vocab_clean = True
    if len(all_pieces) != len(set(all_pieces)):
        failures.append("Duplicate vocabulary pieces detected.")
        vocab_clean = False
    else:
        print("    OK -- all vocabulary pieces are unique")

    if not EXPECTED_CUSTOM_TOKENS:
        print("    No custom tokens expected.")
    else:
        for tok in EXPECTED_CUSTOM_TOKENS:
            found = tok in all_pieces
            print(f"    {tok!r} present as a single piece: {found}")
            if not found:
                failures.append(f"Custom token {tok!r} not found as a single vocab piece")

    # ---- Check 4: no unused/padding-type pieces should exist ----
    print(f"\n[4] Checking for any UNUSED-type placeholder pieces (should be none):")
    unused_like = [p for p in all_pieces if re.fullmatch(r"<unused_\d+>", p)]
    if unused_like:
        print(f"    Found {len(unused_like)} suspicious placeholder-like piece(s): {unused_like[:10]}")
        failures.append(f"Found unexpected placeholder-like pieces (padding was NOT supposed to be applied): {unused_like[:10]}")
    else:
        print(f"    OK -- no placeholder/padding pieces found, matches the no-padding decision")

    # ---- Check 5: round-trip integrity ----
    print(f"\n[5] Round-trip integrity check on serialized QA probes:")
    round_trip_ok = True
    for text in PROBE_SERIALIZED_QA:
        ids = sp.EncodeAsIds(text)
        decoded = sp.DecodeIds(ids)
        ids2 = sp.EncodeAsIds(decoded)
        ok = (decoded == text) and (ids == ids2)
        status = "OK" if ok else "MISMATCH"
        print(f"    [{status}] {text[:50]!r}...")
        if decoded != text:
            failures.append(f"Round-trip mismatch for: {text!r} -> decoded: {decoded!r}")
            round_trip_ok = False
        if ids != ids2:
            failures.append(f"Encoding not stable after round-trip for: {text[:50]!r}")
            round_trip_ok = False
    categories["Round Trip"] = PASS if round_trip_ok else FAIL

    # ---- Check 6: zero <unk> on real text ----
    print(f"\n[6] UNK token check on real probe text:")
    unk_id = sp.unk_id()
    total_unk = sum(sp.EncodeAsIds(t).count(unk_id) for t in PROBE_SERIALIZED_QA)
    print(f"    Total <unk> occurrences across probe texts: {total_unk}")
    byte_fallback_ok = True
    if total_unk > 0:
        failures.append(f"{total_unk} <unk> token(s) found -- byte_fallback may not be working")
        byte_fallback_ok = False
    else:
        print("    OK -- zero <unk> tokens (byte_fallback working as expected)")

    # ---- Check 7: digit splitting + numeric / scientific-notation round-trip ----
    print(f"\n[7] Digit splitting:")
    digit_pieces = sp.EncodeAsPieces("123456789")
    print(f"    {digit_pieces}")
    decoded_digits = sp.DecodePieces(digit_pieces)
    digit_ok = True
    if decoded_digits != "123456789":
        failures.append(f"Digit reconstruction failed: {decoded_digits!r}")
        digit_ok = False
    if len(digit_pieces) == 1:
        failures.append("Digits were not split -- split_digits=True does not appear to be in effect")
        digit_ok = False
    if digit_ok:
        print("    OK -- digits reconstruct correctly and are split into multiple pieces")
    categories["Digit Splitting"] = PASS if digit_ok else FAIL

    print(f"\n    Extended numeric/scientific-notation probes (physics-typical):")
    for np_text in PROBE_NUMERICS:
        ids = sp.EncodeAsIds(np_text)
        decoded = sp.DecodeIds(ids)
        pieces = sp.EncodeAsPieces(np_text)
        ok = nfkc(decoded) == nfkc(np_text)
        status = "OK" if ok else "MISMATCH"
        print(f"    [{status}] {np_text!r} -> {pieces}")
        if not ok:
            failures.append(f"Numeric round-trip mismatch for: {np_text!r} -> decoded: {decoded!r}")

    # ---- Check 8: physics & math symbol round-trip ----
    print(f"\n[8] Physics & Mathematics symbol round-trip (Unicode preservation, NFKC-normalized comparison):")
    symbol_failures = []
    for sym in PROBE_SYMBOLS:
        ids = sp.EncodeAsIds(sym)
        decoded = sp.DecodeIds(ids)
        ok = nfkc(decoded) == nfkc(sym)
        if not ok:
            failures.append(f"Failed physics/math symbol round-trip: {sym!r}")
            symbol_failures.append(sym)
        print(f"    [{'OK' if ok else 'MISMATCH'}] {sym!r} -> {sp.EncodeAsPieces(sym)}")
    if not symbol_failures:
        print(f"    OK -- all {len(PROBE_SYMBOLS)} physics/math symbols round-trip correctly")
    categories["Unicode Physics Symbols"] = PASS if not symbol_failures else FAIL

    # ---- Check 9: NFKC normalization actually applied ----
    print(f"\n[9] NFKC normalization check:")
    nfkc_ok = True
    for raw, expected_norm in PROBE_NFKC:
        ids = sp.EncodeAsIds(raw)
        decoded = sp.DecodeIds(ids)
        ok = decoded == expected_norm
        status = "OK" if ok else "MISMATCH"
        print(f"    [{status}] {raw!r} -> decoded {decoded!r} (expected {expected_norm!r})")
        if not ok:
            nfkc_ok = False
            failures.append(
                f"NFKC normalization not applied as expected: {raw!r} -> {decoded!r}, "
                f"expected {expected_norm!r}"
            )
    categories["NFKC Normalization"] = PASS if nfkc_ok else FAIL

    # ---- Check 10: byte fallback verified directly ----
    print(f"\n[10] Byte-fallback direct verification (rare/emoji input):")
    rare = PROBE_RARE
    rare_pieces = sp.EncodeAsPieces(rare)
    print(f"    pieces: {rare_pieces}")
    has_byte_pieces = any(byte_piece_re.match(p) for p in rare_pieces)
    has_unk = any(p == sp.IdToPiece(unk_id) for p in rare_pieces)
    if has_byte_pieces and not has_unk:
        print("    OK -- rare input decomposed into <0xNN> byte pieces, no <unk> used")
    elif has_unk:
        failures.append("Byte fallback not working -- rare input fell back to <unk> instead of byte pieces")
        print("    FAIL -- <unk> used instead of byte-level pieces")
        byte_fallback_ok = False
    else:
        failures.append("Byte fallback check inconclusive -- no byte pieces and no <unk> found")
        print("    WARNING -- no byte pieces or <unk> detected, inspect manually")
        byte_fallback_ok = False
    rare_decoded = sp.DecodeIds(sp.EncodeAsIds(rare))
    if rare_decoded != rare:
        failures.append(f"Byte-fallback round-trip failed for {rare!r} -> {rare_decoded!r}")
        byte_fallback_ok = False
    else:
        print(f"    OK -- byte-fallback round-trip reconstructs {rare!r} exactly")
    categories["Byte Fallback"] = PASS if byte_fallback_ok else FAIL

    # ---- Check 11: serialization header pieces ----
    print(f"\n[11] Serialization header tokens (Question:/Options:/Answer:):")
    for h in PROBE_HEADERS:
        pieces = sp.EncodeAsPieces(h)
        has_unk_here = unk_id in sp.EncodeAsIds(h)
        print(f"    {h!r} -> {pieces}")
        if has_unk_here:
            failures.append(f"Header {h!r} contains <unk> piece(s)")
    headers_ok = not any(unk_id in sp.EncodeAsIds(h) for h in PROBE_HEADERS)
    if headers_ok:
        print("    OK -- no header text falls back to <unk>")
    categories["Serialization Headers"] = PASS if headers_ok else FAIL

    # ---- Check 12: case sensitivity ----
    print(f"\n[12] Case sensitivity (lowercase=False expected):")
    case_ok = True
    for upper, lower in PROBE_CASE_PAIRS:
        ids_u = sp.EncodeAsIds(upper)
        ids_l = sp.EncodeAsIds(lower)
        differ = ids_u != ids_l
        print(f"    {upper!r} -> {ids_u}   {lower!r} -> {ids_l}   {'OK (differ)' if differ else 'FAIL (same)'}")
        if not differ:
            case_ok = False
            failures.append(f"Case not distinguished: {upper!r} and {lower!r} encode identically")
    categories["Case Sensitivity"] = PASS if case_ok else FAIL

    # ---- Check 13: important physics units round-trip ----
    print(f"\n[13] Physics unit round-trip (NFKC-normalized comparison):")
    units_ok = True
    for u in PROBE_UNITS:
        ids = sp.EncodeAsIds(u)
        decoded = sp.DecodeIds(ids)
        ok = nfkc(decoded) == nfkc(u)
        print(f"    [{'OK' if ok else 'MISMATCH'}] {u!r} -> {sp.EncodeAsPieces(u)}")
        if not ok:
            units_ok = False
            failures.append(f"Unit round-trip failed for {u!r} -> decoded {decoded!r}")
    categories["Physics Units"] = PASS if units_ok else FAIL

    # ---- Check 14: scan vocabulary for junk pieces ----
    print(f"\n[14] Scanning vocabulary for invisible/junk pieces:")
    JUNK_CHARS = "\u200b\u200c\u200d\ufeff\u200e\u200f\u2060\u00ad\ufffd"
    # Byte-fallback pieces (e.g. "<0xE2>") are valid, expected vocabulary
    # entries -- they are not junk, so they're excluded from this scan even
    # though their literal characters ("<", "0", "x", ...) would never match
    # JUNK_CHARS anyway. This filter guards against byte-fallback pieces
    # that happen to decode/display oddly being mistaken for junk.
    non_byte_pieces = [p for p in all_pieces if not byte_piece_re.match(p)]
    junk_pieces = [p for p in non_byte_pieces if any(ch in p for ch in JUNK_CHARS)]
    if junk_pieces:
        print(f"    Found {len(junk_pieces)} junk piece(s): {junk_pieces[:10]}")
        failures.append(f"Vocabulary contains invisible/replacement-char pieces: {junk_pieces[:10]}")
        vocab_clean = False
    else:
        print("    OK -- no invisible/zero-width/replacement-char pieces found in vocabulary "
              "(byte-fallback pieces excluded from this scan)")
    categories["Vocabulary Cleanliness"] = PASS if vocab_clean else FAIL

    # ---- Check 15: performance benchmark (informational only) ----
    N_BENCH = 5000
    print(f"\n[15] Performance benchmark ({N_BENCH} encodes / {N_BENCH} decodes):")
    bench_text = PROBE_SERIALIZED_QA[0]
    bench_ids = sp.EncodeAsIds(bench_text)
    n_bench_tokens = len(bench_ids)

    t0 = time.perf_counter()
    for _ in range(N_BENCH):
        sp.EncodeAsIds(bench_text)
    encode_seconds_total = time.perf_counter() - t0
    encode_ms_per_call = (encode_seconds_total / N_BENCH) * 1000
    encode_tokens_per_second = (n_bench_tokens * N_BENCH) / encode_seconds_total if encode_seconds_total > 0 else float("inf")

    t0 = time.perf_counter()
    for _ in range(N_BENCH):
        sp.DecodeIds(bench_ids)
    decode_seconds_total = time.perf_counter() - t0
    decode_ms_per_call = (decode_seconds_total / N_BENCH) * 1000
    decode_tokens_per_second = (n_bench_tokens * N_BENCH) / decode_seconds_total if decode_seconds_total > 0 else float("inf")

    print(f"    {N_BENCH} encodes: {encode_seconds_total*1000:.1f} ms total, {encode_ms_per_call:.4f} ms/call, "
          f"{encode_tokens_per_second:,.0f} tokens/sec")
    print(f"    {N_BENCH} decodes: {decode_seconds_total*1000:.1f} ms total, {decode_ms_per_call:.4f} ms/call, "
          f"{decode_tokens_per_second:,.0f} tokens/sec")
    report["performance"] = {
        "iterations": N_BENCH,
        "encode_ms_per_call": encode_ms_per_call,
        "decode_ms_per_call": decode_ms_per_call,
        "encode_tokens_per_second": encode_tokens_per_second,
        "decode_tokens_per_second": decode_tokens_per_second,
    }
    categories["Performance Benchmark"] = INFO  # informational; no pass/fail threshold

    # ---- Checks 16-19: held-out corpus stats (vocab utilization, context,
    #      fertility, chars/token) -- only run if a corpus file is given ----
    if corpus_path:
        print(f"\n[16-19] Held-out corpus statistics ({corpus_path}):")

        # Validate the corpus is UTF-8 before doing anything else with it.
        corpus_valid_utf8 = True
        try:
            with open(corpus_path, "rb") as f:
                raw_bytes = f.read()
            raw = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            corpus_valid_utf8 = False
            print(f"    Corpus file is not valid UTF-8: {e}")
            print("    Skipping held-out checks -- re-save the corpus as UTF-8 and re-run.")
            categories["Vocabulary Utilization"] = SKIPPED
            categories["Context Statistics"] = SKIPPED
        except OSError as e:
            corpus_valid_utf8 = False
            print(f"    Could not read corpus file: {e}")
            categories["Vocabulary Utilization"] = SKIPPED
            categories["Context Statistics"] = SKIPPED

        if corpus_valid_utf8:
            samples = [s for s in raw.split("\n\n") if s.strip()]
            if not samples:
                print("    Corpus file appears empty, skipping held-out checks.")
                categories["Vocabulary Utilization"] = SKIPPED
                categories["Context Statistics"] = SKIPPED
            else:
                used_ids = set()
                tok_lens = []
                n_words = 0
                n_chars = 0
                n_toks = 0
                for s in samples:
                    ids = sp.EncodeAsIds(s)
                    used_ids.update(ids)
                    tok_lens.append(len(ids))
                    n_toks += len(ids)
                    n_words += len(word_tokens(s))
                    n_chars += len(s)

                # [16] vocabulary utilization -- informational, no fixed pass/fail
                # threshold (a legitimately small or specialized corpus will
                # naturally exercise far less than the full vocabulary).
                used_count = len(used_ids)
                coverage = 100 * used_count / actual_size
                print(f"    [16] Used vocabulary: {used_count:,} / {actual_size:,}  "
                      f"({coverage:.1f}%)")
                print(f"         Unused pieces: {actual_size - used_count:,}")
                report["vocab_utilization"] = {
                    "used": used_count, "total": actual_size, "coverage_pct": coverage
                }
                categories["Vocabulary Utilization"] = INFO

                # [17] context statistics -- informational. Some long-tail
                # samples legitimately exceeding EXPECTED_CONTEXT is not, by
                # itself, a tokenizer defect (it's a data/packing decision), so
                # this no longer flips the overall verdict to FAIL.
                print(f"    [17] Token-length stats: "
                      f"median={pct(tok_lens,50)}  p95={pct(tok_lens,95)}  "
                      f"p99={pct(tok_lens,99)}  max={max(tok_lens)}")
                n_over_context = sum(1 for l in tok_lens if l > EXPECTED_CONTEXT)
                print(f"         Samples exceeding {EXPECTED_CONTEXT}-token context: "
                      f"{n_over_context:,} / {len(samples):,} "
                      f"({100*n_over_context/len(samples):.2f}%)")
                report["context_stats"] = {
                    "median": pct(tok_lens, 50), "p95": pct(tok_lens, 95),
                    "p99": pct(tok_lens, 99), "max": max(tok_lens),
                    "over_context_count": n_over_context,
                }
                categories["Context Statistics"] = INFO

                # [18] fertility
                fertility = n_toks / max(1, n_words)
                print(f"    [18] Fertility (tokens/word): {fertility:.2f}")
                report["fertility"] = fertility

                # [19] chars per token
                chars_per_tok = n_chars / max(1, n_toks)
                print(f"    [19] Chars/token: {chars_per_tok:.2f}")
                report["chars_per_token"] = chars_per_tok
    else:
        print(f"\n[16-19] Held-out corpus statistics: SKIPPED (no corpus file provided)")
        categories["Vocabulary Utilization"] = SKIPPED
        categories["Context Statistics"] = SKIPPED

    categories["Byte Fallback"] = PASS if byte_fallback_ok else FAIL

    report["digit_split"] = len(digit_pieces) > 1
    report["passed"] = len(failures) == 0
    report["failures"] = failures
    report["categories"] = categories

    print("\n" + "=" * 70)
    print("MODEL300M TOKENIZER VERIFICATION SUMMARY")
    print("=" * 70)
    category_order = [
        "Vocabulary Size", "Special Tokens", "Round Trip", "Digit Splitting",
        "Byte Fallback", "Unicode Physics Symbols", "Physics Units", "NFKC Normalization",
        "Case Sensitivity", "Serialization Headers", "Vocabulary Cleanliness",
        "Vocabulary Utilization", "Context Statistics", "Performance Benchmark",
    ]
    label_width = max(len(name) for name in category_order)
    for name in category_order:
        label = categories.get(name, SKIPPED)
        print(f"{name:<{label_width}} : {label}")
    overall = "PASS" if not failures else "FAIL"
    print()
    print(f"{'Overall Status':<{label_width}} : {overall}")
    print("(PASS/FAIL categories determine overall status; INFO/SKIPPED do not.)")
    print("=" * 70)

    if failures:
        print(f"\n{len(failures)} issue(s) found:")
        for f in failures:
            print(f"  - {f}")
        print()

    report_path = os.path.splitext(model_path)[0] + "_verification.json"
    with io.open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Full report written to {report_path}")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()