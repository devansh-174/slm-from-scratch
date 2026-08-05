# Utility — Tokenizer Verification

## Overview

Verifies the trained SentencePiece tokenizer independently after training.

The script validates vocabulary integrity, special tokens, encoding/decoding behavior, and tokenizer consistency before it is used for dataset preparation and model training.

---

## Features

- Vocabulary verification
- Special token validation
- Custom token verification
- Round-trip encoding check
- UNK token verification
- Padding verification
- Tokenizer integrity checks

---

## Workflow

```text
Tokenizer Model
      │
      ▼
Vocabulary Validation
      │
      ▼
Special Token Check
      │
      ▼
Encoding Validation
      │
      ▼
Tokenizer Verification
```

---

## Output

- Tokenizer verification report

---

## Outcome

Confirms that the production tokenizer is correctly trained and ready for downstream dataset preparation and model training.