# `make_bins1.py`

Converts the training and validation JSONL datasets into binary token-ID files using the production SentencePiece tokenizer.

## Configuration

| Component | Value |
|---|---|
| Training data | `train1.jsonl` |
| Validation data | `validation1.jsonl` |
| Tokenizer | `tokenizer/slm_tokenizer_11000.model` |
| Training output | `train1.bin` |
| Validation output | `val1.bin` |
| Validation fraction | 10% |
| Random seed | 42 |
| Token dtype | `uint16` |

## Usage

```bash
python3 make_bins1.py