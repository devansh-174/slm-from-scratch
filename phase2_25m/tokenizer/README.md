# Tokenizer — Phase 2 (~25.7M SLM)

Production tokenizer pipeline for Phase 2 of the SLM.

```text id="7p4v8n"
train1.jsonl
     │
     ▼
tokenizer_train.py
     │
     ├── slm_tokenizer_11000.model
     └── slm_tokenizer_11000.vocab
              │
              ▼
     tokenizer_verify.py
              │
              ▼
          VERIFIED
```

## Files

### `tokenizer_train.py`

Trains the final **SentencePiece Unigram** tokenizer from `train1.jsonl`.

### `tokenizer_verify.py`

Independently loads the generated `.model` file and verifies that the tokenizer is internally consistent and matches the expected production configuration.

## Configuration

| Setting               | Value                 |
| --------------------- | --------------------- |
| Tokenizer             | SentencePiece Unigram |
| Target vocabulary     | `11,000`              |
| Context length        | `1,400`               |
| Character coverage    | `1.0`                 |
| Normalization         | NFKC                  |
| Lowercasing           | OFF                   |
| Byte fallback         | ON                    |
| Digit splitting       | ON                    |
| Custom tokens         | None                  |
| UNK / BOS / EOS / PAD | `0 / 1 / 2 / 3`       |
| Vocabulary padding    | OFF                   |
| Placeholder cleaning  | ON                    |

### Vocabulary

The target vocabulary is `11,000`. No post-hoc padding is applied. If the corpus cannot naturally support the requested size, the trainer uses the maximum vocabulary supported by SentencePiece.

### Placeholder Cleaning

Runs of 3+ non-alphanumeric, non-whitespace characters such as:

```text
****
.....
-----
*******
```

are collapsed to the canonical:

```text
___
```

before tokenizer training.

## Input

```text
train1.jsonl
```

The trainer serializes:

```text
question
options
answer
solution
```

into the tokenizer-training corpus.

## Output

```text
tokenizer/
├── train_corpus.txt
├── slm_tokenizer_11000.model
└── slm_tokenizer_11000.vocab
```

The `.model` file is the tokenizer used for subsequent SLM training.

## Usage

Train:

```bash
python3 tokenizer_train.py
```

Verify:

## Verification Checks

`tokenizer_verify.py` independently checks:

* Actual vocabulary size
* UNK/BOS/EOS/PAD IDs
* Custom-token absence
* Padding/`<unused_N>` absence
* Encode/decode round-trip integrity
* Encoding stability
* `<unk>` occurrences on probe text
* Digit splitting

A successful verification ends with:

```text
VERIFICATION PASSED
```

The tokenizer should be verified successfully before being used for Phase-2 model training.
