The model is trained on a physics-focused corpus and fine-tuned on physics-specific examples. It is also evaluated on broader mathematics, science, and knowledge benchmarks, including topics beyond its training domain.

## Model architecture

| Component | Configuration |
|---|---:|
| Architecture | Decoder-only Transformer |
| Parameters | Approximately 300 million |
| Hidden size | 1,536 |
| Transformer layers | 10 |
| Query attention heads | 24 |
| Key/value heads | 6 |
| Head dimension | 64 |
| Context length | 2,048 tokens |
| FFN intermediate size | 4,864 |
| Normalization | RMSNorm |
| Positional encoding | RoPE |
| FFN | SwiGLU / SiLU |
| Embedding/output weights | Tied |

The model uses grouped-query attention (GQA).

## Data and tokenizer

- Pretraining corpus: approximately **100 million tokens**, as reported.
- Fine-tuning data: physics-specific.
- Tokenizer: SentencePiece Unigram, vocabulary size **11,000** in the reviewed configuration.
- Recorded tokenizer settings include NFKC normalization, byte fallback, digit splitting, and full character coverage.

## Pipeline

1. Train and verify the tokenizer.
2. Prepare and serialize datasets; create tokenized training and validation binaries.
3. Pretrain the decoder-only model.
4. Fine-tune the pretrained checkpoint on physics-specific examples.
5. Generate predictions using the inference scripts.
6. Evaluate existing predictions against ground truth and benchmark datasets.

## Checkpoints compared

The supplied experiments compare variants named **11.5K** and **14K**.

## Results

### External benchmark suite

| Benchmark | Metric | 11.5K | 14K | Higher score |
|---|---|---:|---:|---|
| GSM8K | Numeric accuracy | 1.90% | 2.05% | 14K |
| MATH-500 | Numeric accuracy | 6.40% | 6.00% | 11.5K |
| MMLU | Accuracy | 10.86% | 10.53% | 11.5K |
| SciQ | Accuracy | 0.90% | 0.90% | Tie |
| ARC Challenge | Accuracy | 0.00% | 0.00% | Tie |
| Macro average accuracy | Average across benchmarks | 4.01% | 3.90% | 11.5K |
| Micro average accuracy | Aggregate accuracy | 9.26% | 8.99% | 11.5K |
| Micro MCQ accuracy | Multiple-choice benchmarks | 9.99% | 9.69% | 11.5K |
| Micro numeric accuracy | Numeric benchmarks | 3.14% | 3.14% | Tie |
| Coverage | Evaluated samples | 100% | 100% | Tie |


### Physics evaluation on the held-out test dataset

| Metric | 11.5K | 14K | Higher score |
|---|---:|---:|---|
| Final answer accuracy | 1.654% | 1.399% | 11.5K |
| Numeric accuracy | 1.654% | 1.399% | 11.5K |
| Formula accuracy | 58.02% | 57.89% | 11.5K |
| Substitution accuracy | 58.14% | 58.14% | Tie |
| Unit accuracy | 6.62% | 6.74% | 14K |
| ROUGE-L | 0.0256 | 0.0257 | 14K |
| Average inference time | 1.107 s | 1.138 s | 11.5K |
| Average generated tokens | 145.15 | 147.46 | — |


### Physics evaluation on generated data from class nusery to class 10 physics


| Metric | 11.5K | 14K | Higher score |
|---|---:|---:|---|
| Final answer accuracy | 32.80% | 32.26% | 11.5K |
| Numeric accuracy | 32.62% | 32.10% | 11.5K |
| Formula accuracy | 59.66% | 67.55% | 14K |
| Substitution accuracy | 60.86% | 84.83% | 14K |
| Unit accuracy | 38.32% | 38.12% | 11.5K |
| ROUGE-L | 0.1980 | 0.1920 | 11.5K |



## Metrics and interpretation

- **Final-answer / numeric accuracy:** correctness under the evaluator's extraction and tolerance rules.
- **Formula / substitution accuracy:** normalized string comparisons; these do not prove symbolic equivalence or validate the reasoning.
- **BLEU / ROUGE:** text-overlap metrics, not factual correctness measures.
- **Coverage:** proportion of benchmark examples evaluated.
- **Latency / token counts:** generation statistics under the reported setup.

