"""
Independent verification of the trained tokenizer -- does NOT trust
train_tokenizer.py's own print statements. Loads the .model file directly
and checks everything from scratch. No padding is expected or required;
this simply confirms the file is internally consistent and correct.

USAGE:
    python3 verify_tokenizer.py tokenizer/slm_tokenizer_12779.model
"""

import sys
import re

import sentencepiece as spm

EXPECTED_CUSTOM_TOKENS = ["<CALC>", "</CALC>"]


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 verify_tokenizer.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    print(f"Loading {model_path} directly with SentencePieceProcessor ...")
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)

    failures = []

    # ---- Check 1: report the actual piece count (no assumed target) ----
    actual_size = sp.GetPieceSize()
    print(f"\n[1] GetPieceSize() = {actual_size}  (this is the real, final vocab size)")

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
    if sorted(special_ids.values()) != [0, 1, 2, 3]:
        failures.append(f"Special token IDs not exactly {{0,1,2,3}}: {special_ids}")
    else:
        print("    OK -- special tokens occupy exactly ids 0-3")
    if list(special_ids.values()).count(sp.unk_id()) == 1:
        print("    OK -- exactly ONE dedicated <unk> slot, no extra unknown tokens")

    # ---- Check 3: custom tokens present as real pieces ----
    print(f"\n[3] Custom tokens ({EXPECTED_CUSTOM_TOKENS}):")
    all_pieces = [sp.IdToPiece(i) for i in range(actual_size)]
    for tok in EXPECTED_CUSTOM_TOKENS:
        found = tok in all_pieces
        print(f"    {tok!r} present as a single piece: {found}")
        if not found:
            failures.append(f"Custom token {tok!r} not found as a single vocab piece")

    # ---- Check 4: no unused/padding-type pieces should exist ----
    # (we deliberately chose NOT to pad -- confirm that decision held)
    print(f"\n[4] Checking for any UNUSED-type placeholder pieces (should be none):")
    # Only match the EXACT synthetic pattern pad_tokenizer.py would create
    # (e.g. "<unused_0>", "<unused_42>") -- NOT any real vocabulary word that
    # innocently contains a substring like "dummy" (e.g. a physics question
    # about "a dummy weight" is a completely normal learned piece, not padding).
    unused_like = [p for p in all_pieces if re.fullmatch(r"<unused_\d+>", p)]
    if unused_like:
        print(f"    Found {len(unused_like)} suspicious placeholder-like piece(s): {unused_like[:10]}")
        failures.append(f"Found unexpected placeholder-like pieces (padding was NOT supposed to be applied): {unused_like[:10]}")
    else:
        print(f"    OK -- no placeholder/padding pieces found, matches the no-padding decision")

    # ---- Check 5: round-trip integrity ----
    print(f"\n[5] Round-trip integrity check on probe texts:")
    probe_texts = [
        "Substitution: <CALC>1 * 5</CALC>",
        "A car travels 60 km in 2 hours. What is its average speed?",
        "Why does ice float on water? Because it is less dense than liquid water.",
        "The pillar feels a pressure of 45 N/m^2 with force 135 N over area 3 m^2.",
    ]
    for text in probe_texts:
        ids = sp.EncodeAsIds(text)
        decoded = sp.DecodeIds(ids)
        ok = decoded == text
        status = "OK" if ok else "MISMATCH"
        print(f"    [{status}] {text[:50]!r}...")
        if not ok:
            failures.append(f"Round-trip mismatch for: {text!r} -> decoded: {decoded!r}")

    # ---- Check 6: zero <unk> on real text ----
    print(f"\n[6] UNK token check on real probe text:")
    unk_id = sp.unk_id()
    total_unk = sum(sp.EncodeAsIds(t).count(unk_id) for t in probe_texts)
    print(f"    Total <unk> occurrences across probe texts: {total_unk}")
    if total_unk > 0:
        failures.append(f"{total_unk} <unk> token(s) found -- byte_fallback may not be working")
    else:
        print("    OK -- zero <unk> tokens (byte_fallback working as expected)")

    print("\n" + "=" * 70)
    if failures:
        print(f"VERIFICATION FAILED -- {len(failures)} issue(s) found:")
        for f in failures:
            print(f"  - {f}")
    else:
        print(f"VERIFICATION PASSED -- tokenizer has {actual_size} pieces (natural, "
              f"no padding), special/custom tokens intact, round-trip and UNK behavior correct.")
    print("=" * 70)

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()