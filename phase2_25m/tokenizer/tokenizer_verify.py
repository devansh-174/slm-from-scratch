"""
Independent verification of the trained tokenizer -- does NOT trust
train_tokenizer.py's own print statements. Loads the .model file directly
and checks everything from scratch. No padding is expected or required;
this simply confirms the file is internally consistent and correct.

"""

import sys
import re

import sentencepiece as spm

EXPECTED_CUSTOM_TOKENS = []
EXPECTED_VOCAB = 11000


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 verify_tokenizer.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    print(f"Loading {model_path} directly with SentencePieceProcessor ...")
    sp = spm.SentencePieceProcessor()
    try:
        if not sp.Load(model_path):
            print(f"Failed to load tokenizer: {model_path}")
            sys.exit(1)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        sys.exit(1)

    failures = []

    # ---- Check 1: report the actual piece count and verify against expected ----
    actual_size = sp.GetPieceSize()
    print(f"\n[1] GetPieceSize() = {actual_size}  (this is the real, final vocab size)")
    if actual_size != EXPECTED_VOCAB:
        failures.append(f"Vocabulary mismatch. Expected {EXPECTED_VOCAB}, got {actual_size}")
    else:
        print(f"    OK -- vocabulary size = {EXPECTED_VOCAB}")

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
    if sp.unk_id() < 0:
        failures.append("Missing UNK token")
    if sp.bos_id() < 0:
        failures.append("Missing BOS token")
    if sp.eos_id() < 0:
        failures.append("Missing EOS token")
    if sp.pad_id() < 0:
        failures.append("Missing PAD token")
    if all(tid >= 0 for tid in special_ids.values()):
        print("    OK -- unk/bos/eos/pad are all present and valid")
    if list(special_ids.values()).count(sp.unk_id()) == 1:
        print("    OK -- exactly ONE dedicated <unk> slot, no extra unknown tokens")

    # ---- Check 3: custom tokens present as real pieces ----
    print(f"\n[3] Custom tokens:")
    if not EXPECTED_CUSTOM_TOKENS:
        print("    None expected.")
    else:
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
    all_pieces = [sp.IdToPiece(i) for i in range(actual_size)]
    unused_like = [p for p in all_pieces if re.fullmatch(r"<unused_\d+>", p)]
    if unused_like:
        print(f"    Found {len(unused_like)} suspicious placeholder-like piece(s): {unused_like[:10]}")
        failures.append(f"Found unexpected placeholder-like pieces (padding was NOT supposed to be applied): {unused_like[:10]}")
    else:
        print(f"    OK -- no placeholder/padding pieces found, matches the no-padding decision")

    # ---- Check 5: round-trip integrity ----
    print(f"\n[5] Round-trip integrity check on probe texts:")
    probe_texts = [
        """Question: The magnetic force on a moving charged particle can change the particle's

Options:
A) speed
B) direction
C) Both of these
D) Neither of these

Answer: B""",
        """Question: A car travels 60 km in 2 hours. What is its average speed?

Answer: 30 km/h

Solution: Speed = Distance / Time = 30 km/h""",
        """Question: Why does ice float on water?

Solution: Ice has lower density than liquid water.""",
        """Question: What is the SI unit of force?

Answer: Newton""",
    ]
    for text in probe_texts:
        ids = sp.EncodeAsIds(text)
        decoded = sp.DecodeIds(ids)
        ids2 = sp.EncodeAsIds(decoded)
        ok = (decoded == text) and (ids == ids2)
        status = "OK" if ok else "MISMATCH"
        print(f"    [{status}] {text[:50]!r}...")
        if decoded != text:
            failures.append(f"Round-trip mismatch for: {text!r} -> decoded: {decoded!r}")
        if ids != ids2:
            failures.append(f"Encoding not stable after round-trip for: {text[:50]!r}")

    # ---- Check 6: zero <unk> on real text ----
    print(f"\n[6] UNK token check on real probe text:")
    unk_id = sp.unk_id()
    total_unk = sum(sp.EncodeAsIds(t).count(unk_id) for t in probe_texts)
    print(f"    Total <unk> occurrences across probe texts: {total_unk}")
    if total_unk > 0:
        failures.append(f"{total_unk} <unk> token(s) found -- byte_fallback may not be working")
    else:
        print("    OK -- zero <unk> tokens (byte_fallback working as expected)")

    # ---- Check 7: digit splitting ----
    print(f"\n[7] Digit splitting:")
    digit_pieces = sp.EncodeAsPieces("123456789")
    print(f"    {digit_pieces}")
    decoded_digits = sp.DecodePieces(digit_pieces)
    if decoded_digits != "123456789":
        failures.append(f"Digit reconstruction failed: {decoded_digits!r}")
    if len(digit_pieces) == 1:
        failures.append("Digits were not split -- split_digits=True does not appear to be in effect")
    if decoded_digits == "123456789" and len(digit_pieces) > 1:
        print("    OK -- digits reconstruct correctly and are split into multiple pieces")

    print("\n" + "=" * 70)
    if failures:
        print(f"VERIFICATION FAILED -- {len(failures)} issue(s) found:")
        for f in failures:
            print(f"  - {f}")
    else:
        print(
            f"VERIFICATION PASSED -- tokenizer has {actual_size} pieces, "
            f"vocabulary matches expected size, no padding placeholders, "
            f"round-trip integrity verified, encoding stable, "
            f"digit splitting verified, and zero <unk> tokens observed."
        )
    print("=" * 70)

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()