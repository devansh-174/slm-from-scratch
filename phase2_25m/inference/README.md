# `inference1.py`

Inference script for the **Phase 2 (~25.7M parameter) Physics SLM**.

## Configuration

| Component | Value |
|---|---|
| Model | `PhysicsSLM` |
| Model definition | `model1.py` |
| Vocabulary | 11,000 |
| Tokenizer | `tokenizer/slm_tokenizer_11000.model` |
| Default checkpoint | `checkpoints/best_model.pt` |
| Context length | Read from checkpoint |

## Usage

### Interactive

```bash
python3 inference1.py