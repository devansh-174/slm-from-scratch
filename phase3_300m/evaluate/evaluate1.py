"""
Standalone evaluation script for the Physics SLM (PhysicsSLM / model2.py).

PURE EVALUATION MODE

This version does NOT run inference. It consumes two already-generated
prediction JSONL files (produced separately, e.g. via inference_fine_tune.py)
and scores them against a held-out ground-truth JSONL, producing per-sample
metrics, grouped metrics, and a final markdown comparison report.

Usage
    python evaluate_physics_slm.py \\
        --test-file "/mnt/hdd-data4/interns/slm from scratch/training_data_clean.jsonl" \\
        --preds-11k "/mnt/hdd-data4/interns/slm from scratch/benchmark_predictions_training_data_11500_new.jsonl" \\
        --preds-14k "/mnt/hdd-data4/interns/slm from scratch/benchmark_predictions_training_data_14000.jsonl" \\
        --out-dir eval_results \\
        --workers 8

IMPORTANT
    No model loading, no tokenizer, no --prompt-file: predictions already
    exist, so the SFT prompt template is irrelevant here. If you actually
    need to (re)generate predictions, that is a separate inference script,
    not this one.

    Ground truth (--test-file) and predictions are paired strictly by id.
    Ids present in one file but not the other are reported and skipped, not
    silently dropped without a trace.

Notes
    - Answer parsing has a fallback path: if the model output doesn't emit the
      Given/Formula/Substitution/Final Answer field labels, the script still
      extracts a best-effort final answer instead of silently scoring zero.
    - Formula/substitution comparison is normalized (whitespace, case, common
      operator variants) rather than requiring byte-identical strings.
    - BERTScore downloads roberta-large on first run (needs internet).
    - Output filenames are suffixed with "(1)" to keep this run's outputs
      separate from any prior evaluation run in the same --out-dir.
"""

import argparse
import csv
import json
import multiprocessing as mp
import re
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Physics SLM checkpoints from existing prediction files.")
    p.add_argument("--test-file", required=True, help="Path to ground-truth JSONL (training_data_clean.jsonl)")
    p.add_argument("--preds-11k", required=True, help="Path to existing 11.5k prediction JSONL")
    p.add_argument("--preds-14k", required=True, help="Path to existing 14k prediction JSONL")
    p.add_argument("--out-dir", default="eval_results", help="Output directory")
    p.add_argument("--workers", type=int, default=8, help="Worker processes for metric scoring")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only first N samples (debug)")
    p.add_argument("--skip-bertscore", action="store_true", help="Skip BERTScore (faster, no download)")
    return p.parse_args()


# ==============================================================================
# I/O helpers
# ==============================================================================

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_predictions(path: str) -> dict:
    """
    Load an existing predictions JSONL into {id: prediction_text}.
    Fails loudly (KeyError) if a row is missing 'id' or a prediction field,
    rather than silently skipping malformed rows.
    """
    rows = load_jsonl(path)
    pred_by_id = {}
    pred_field_candidates = ("prediction", "generated_text", "output", "response")
    for row in rows:
        if "id" not in row:
            raise KeyError(f"{path}: row missing 'id' field: {row}")
        pred_field = next((k for k in pred_field_candidates if k in row), None)
        if pred_field is None:
            raise KeyError(
                f"{path}: row {row.get('id')} has none of the expected prediction "
                f"fields {pred_field_candidates}; found keys: {list(row.keys())}"
            )
        pred_by_id[row["id"]] = row[pred_field]
    return pred_by_id


# ==============================================================================
# Answer parsing (model output -> structured fields, mirroring answer schema)
# ==============================================================================

FIELD_PATTERNS = {
    "given": r"(?|known)\s*[:-]\s*(.+?)(?=\n[a-z]+\s*[:-]|\Z)",
    "formula": r"formula\s*[:-]\s*(.+?)(?=\n[a-z]+\s*[:-]|\Z)",
    "substitution": r"substitut\w*\s*[:-]\s*(.+?)(?=\n[a-z]+\s*[:-]|\Z)",
    "final_answer": r"final\sanswer\s[:-]\s*(.+?)(?=\n[a-z]+\s*[:-]|\Z)",
}

NUMBER_RE = re.compile(r"-?\d+.?\d*(?:[eE]-?\d+)?")
UNIT_TOKEN_RE = re.compile(r"[a-zA-Z°Ω][a-zA-Z°Ω/²³%^.]*")

# Units the fallback final-answer extractor should recognize when the model
# doesn't label its output at all -- extend this list for your domain.
KNOWN_UNITS = {
    "m", "cm", "mm", "km", "s", "ms", "kg", "g", "n", "j", "w", "pa", "v",
    "a", "ohm", "ω", "hz", "c", "k", "mol", "rad", "deg", "°", "m/s", "m/s2",
    "m/s^2", "kg/m3", "kg/m^3", "j/kg", "w/m2", "n/m", "n/m2", "cal",
}


def _fallback_extract_final_answer(text: str) -> str:
    """
    Used when the model output has no 'Final Answer:' label at all.
    Prefers the last line containing a number; falls back to the last
    non-empty line; falls back to the whole (short) text.
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return ""
    for line in reversed(lines):
        if NUMBER_RE.search(line):
            return line
    return lines[-1]


def parse_structured_answer(text: str) -> dict:
    text_l = text.lower()
    parsed = {"explanation": text.strip()}
    any_field_found = False
    for field, pattern in FIELD_PATTERNS.items():
        m = re.search(pattern, text_l, re.IGNORECASE | re.DOTALL)
        if m:
            span = m.span(1)
            parsed[field] = text[span[0]].strip()
            any_field_found = True
        else:
            parsed[field] = ""

    # Fallback path: the model produced free-form prose instead of the
    # labeled Given/Formula/Substitution/Final Answer schema. Don't silently
    # score every structured metric as zero -- still make a best-effort
    # attempt at a final answer so numeric/unit accuracy remain meaningful.
    if not parsed["final_answer"]:
        parsed["final_answer"] = _fallback_extract_final_answer(text)
        parsed["final_answer_was_fallback"] = not any_field_found
    else:
        parsed["final_answer_was_fallback"] = False

    return parsed


def normalize_expression(s: str) -> str:
    """
    Normalize a formula/substitution string for comparison: strip
    whitespace, lowercase, unify common operator/notation variants.
    This is NOT symbolic equivalence (e.g. it won't know F=ma == a=F/m),
    just tolerant of formatting differences that shouldn't count as errors.
    """
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("×", "").replace("·", "")
    s = s.replace("÷", "/")
    s = s.replace("^", "**")
    return s


def extract_number(s: str):
    if not s:
        return None
    m = NUMBER_RE.search(s.replace(",", ""))
    return float(m.group()) if m else None


def extract_units(s: str):
    """
    Return the set of plausible unit tokens found in a string (not just
    the last token), matched against KNOWN_UNITS where possible.
    """
    if not s:
        return set()
    tokens = [t.lower() for t in UNIT_TOKEN_RE.findall(s)]
    found = {t for t in tokens if t in KNOWN_UNITS}
    if found:
        return found

    # nothing matched the known-unit list; fall back to raw trailing token(s)
    return set(tokens[-1:]) if tokens else set()


# ==============================================================================
# Metrics (computed per-sample in worker processes)
# ==============================================================================

def bleu_n(ref_tokens, hyp_tokens, n):
    from collections import Counter

    def ngrams(tokens, n):
        return Counter(tuple(tokens[i + n]) for i in range(len(tokens) - n + 1)) if len(tokens) >= n else Counter()

    ref_ngrams, hyp_ngrams = ngrams(ref_tokens, n), ngrams(hyp_tokens, n)
    if not hyp_ngrams:
        return 0.0
    overlap = sum((hyp_ngrams & ref_ngrams).values())
    total = sum(hyp_ngrams.values())
    precision = overlap / total if total else 0.0
    bp = 1.0 if len(hyp_tokens) >= len(ref_tokens) else (
        __import__("math").exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1))
    )
    return precision * bp


def rouge_l(ref_tokens, hyp_tokens):
    m, n = len(ref_tokens), len(hyp_tokens)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec = lcs / n
    rec = lcs / m
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def compute_sample_metrics(args):
    sample, pred_text = args
    gt = sample["answer"]
    parsed_pred = parse_structured_answer(pred_text)

    ref_final = str(gt.get("final_answer", ""))
    pred_final = parsed_pred.get("final_answer", "")

    ref_tokens = gt.get("explanation", "").split()
    hyp_tokens = parsed_pred.get("explanation", "").split()

    # NOTE: this is a strict, near-always-zero diagnostic metric -- it only
    # fires when the model reproduces the reference's exact label formatting
    # verbatim. It is intentionally not meant to be read as a primary score;
    # use final_answer_accuracy / numeric_accuracy for that.
    template_exact_match = int(pred_text.strip() == (
        f"Given: {gt.get('given','')}\nFormula: {gt.get('formula','')}\n"
        f"Substitution: {gt.get('substitution','')}\nFinal Answer: {gt.get('final_answer','')}"
    ).strip())

    bleu = {f"bleu{n}": bleu_n(ref_tokens, hyp_tokens, n) for n in range(1, 5)}
    rougeL = rouge_l(ref_tokens, hyp_tokens)

    ref_num = extract_number(ref_final)
    pred_num = extract_number(pred_final)
    numeric_acc = int(ref_num is not None and pred_num is not None and abs(ref_num - pred_num) < 1e-3 * max(abs(ref_num), 1))

    final_answer_acc = int(ref_final.strip().lower() == pred_final.strip().lower()) or numeric_acc

    formula_acc = int(
        normalize_expression(gt.get("formula", "")) == normalize_expression(parsed_pred.get("formula", ""))
    )
    substitution_acc = int(
        normalize_expression(gt.get("substitution", "")) == normalize_expression(parsed_pred.get("substitution", ""))
    )

    ref_units = extract_units(ref_final)
    pred_units = extract_units(pred_final)
    unit_acc = int(bool(ref_units) and bool(ref_units & pred_units))

    explanation_similarity = rouge_l(ref_tokens, hyp_tokens)  # proxy; BERTScore added separately

    return {
        "id": sample["id"],
        "grade_level": sample.get("grade_level"),
        "topic": sample.get("topic"),
        "question_type": sample.get("question_type"),
        "template_exact_match": template_exact_match,
        **bleu,
        "rougeL": rougeL,
        "numeric_accuracy": numeric_acc,
        "final_answer_accuracy": final_answer_acc,
        "formula_accuracy": formula_acc,
        "substitution_accuracy": substitution_acc,
        "unit_accuracy": unit_acc,
        "explanation_similarity": explanation_similarity,
        "answer_was_unstructured": parsed_pred.get("final_answer_was_fallback", False),
        "pred_final_answer": pred_final,
        "ref_final_answer": ref_final,
    }


def compute_bertscore(preds, refs):
    from bert_score import score as bert_score_fn
    P, R, F1 = bert_score_fn(preds, refs, lang="en", model_type="roberta-large", verbose=False)
    return F1.tolist()


# ==============================================================================
# Per-checkpoint evaluation pipeline (scoring only -- no inference)
# ==============================================================================

def evaluate_from_predictions(preds_path, tag, samples, args):
    print(f"\n===== Evaluating checkpoint [{tag}] (from existing predictions) =====")
    pred_by_id = load_predictions(preds_path)
    sample_by_id = {s["id"]: s for s in samples}

    missing_gt = [i for i in pred_by_id if i not in sample_by_id]
    missing_pred = [s["id"] for s in samples if s["id"] not in pred_by_id]
    if missing_gt:
        print(f"[warn] {tag}: {len(missing_gt)} prediction ids have no matching ground-truth sample; skipping them.")
    if missing_pred:
        print(f"[warn] {tag}: {len(missing_pred)} ground-truth ids have no matching prediction; skipping them.")

    pairs = [(sample_by_id[i], pred_by_id[i]) for i in pred_by_id if i in sample_by_id]
    print(f"[info] {tag}: {len(pairs)} matched (ground-truth, prediction) pairs -- scoring, no generation.")

    print(f"[info] Scoring {len(pairs)} samples with {args.workers} workers...")
    with mp.Pool(args.workers) as pool:
        results = list(tqdm(pool.imap(compute_sample_metrics, pairs, chunksize=8), total=len(pairs), desc=f"Scoring [{tag}]"))

    n_unstructured = sum(1 for r in results if r.get("answer_was_unstructured"))
    if n_unstructured:
        pct = 100.0 * n_unstructured / max(len(results), 1)
        print(f"[warn] {tag}: {n_unstructured}/{len(results)} ({pct:.1f}%) predictions had no labeled "
              f"'Final Answer:' field -- final_answer/numeric accuracy used fallback extraction for these.")

    if not args.skip_bertscore:
        print("[info] Computing BERTScore (this may download roberta-large on first run)...")
        preds_text = [pred_by_id[r["id"]] for r in results]
        refs_text = [sample_by_id[r["id"]]["answer"].get("explanation", "") for r in results]
        try:
            bert_f1 = compute_bertscore(preds_text, refs_text)
            for r, f1 in zip(results, bert_f1):
                r["bertscore_f1"] = f1
        except Exception as e:
            print(f"[warn] BERTScore failed ({e}); filling with None.")
            for r in results:
                r["bertscore_f1"] = None
    else:
        for r in results:
            r["bertscore_f1"] = None

    return results


# ==============================================================================
# Aggregation / reporting
# ==============================================================================

METRIC_KEYS = [
    "template_exact_match", "bleu1", "bleu2", "bleu3", "bleu4", "rougeL", "bertscore_f1",
    "numeric_accuracy", "final_answer_accuracy", "formula_accuracy",
    "substitution_accuracy", "unit_accuracy", "explanation_similarity",
]


def aggregate(results):
    agg = {}
    n = len(results)
    for key in METRIC_KEYS:
        vals = [r[key] for r in results if r.get(key) is not None]
        agg[key] = sum(vals) / len(vals) if vals else None
    agg["n_samples"] = n
    agg["n_unstructured_output"] = sum(1 for r in results if r.get("answer_was_unstructured"))
    return agg


def grouped_aggregate(results, group_key):
    groups = defaultdict(list)
    for r in results:
        groups[r.get(group_key)].append(r)
    return {g: aggregate(rs) for g, rs in groups.items()}


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_group_csv(path, grouped: dict, group_col: str):
    fieldnames = [group_col] + METRIC_KEYS + ["n_samples", "n_unstructured_output"]
    rows = []
    for g, agg in grouped.items():
        row = {group_col: g}
        row.update(agg)
        rows.append(row)
    write_csv(path, rows, fieldnames)


def write_report(path, summary_a, summary_b, grouped_a, grouped_b, pred_paths, tags):
    ta, tb = tags
    lines = ["# Physics SLM Evaluation Report\n"]
    lines.append(f"- {ta} predictions: {pred_paths[ta]}")
    lines.append(f"- {tb} predictions: {pred_paths[tb]}")
    lines.append("- Mode: pure evaluation (predictions were not regenerated)\n")

    lines.append("## Overall Metrics\n")
    lines.append(f"| Metric | {ta} | {tb} | Delta ({tb} - {ta}) |")
    lines.append("|---|---|---|---|")
    for key in METRIC_KEYS:
        v1 = summary_a.get(key)
        v2 = summary_b.get(key)
        if v1 is None or v2 is None:
            lines.append(f"| {key} | {v1} | {v2} | - |")
        else:
            lines.append(f"| {key} | {v1:.4f} | {v2:.4f} | {v2 - v1:+.4f} |")
    lines.append(
        f"\nn_samples: {ta}={summary_a['n_samples']}, {tb}={summary_b['n_samples']} | "
        f"unstructured output: {ta}={summary_a['n_unstructured_output']}, {tb}={summary_b['n_unstructured_output']}\n"
    )
    lines.append(
        "> template_exact_match is a strict diagnostic (verbatim match to the reference's exact "
        "label formatting) -- treat final_answer_accuracy / numeric_accuracy as the primary scores.\n"
    )

    for group_name, ga, gb in [("Grade Level", grouped_a["grade_level"], grouped_b["grade_level"]),
                               ("Topic", grouped_a["topic"], grouped_b["topic"]),
                               ("Question Type", grouped_a["question_type"], grouped_b["question_type"])]:
        lines.append(f"## By {group_name}\n")
        lines.append(f"| {group_name} | {ta} final_answer_acc | {tb} final_answer_acc | {ta} rougeL | {tb} rougeL | n |")
        lines.append("|---|---|---|---|---|---|")
        keys = sorted(set(ga.keys()) | set(gb.keys()), key=lambda x: str(x))
        for k in keys:
            a1 = ga.get(k, {})
            a2 = gb.get(k, {})
            lines.append(
                f"| {k} | {a1.get('final_answer_accuracy', 0) or 0:.3f} | {a2.get('final_answer_accuracy', 0) or 0:.3f} "
                f"| {a1.get('rougeL', 0) or 0:.3f} | {a2.get('rougeL', 0) or 0:.3f} | {a1.get('n_samples', 0)} |"
            )
        lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl(args.test_file)
    if args.limit:
        samples = samples[: args.limit]
    print(f"[info] Loaded {len(samples)} ground-truth samples from {args.test_file}")

    tags = ("11k", "14k")
    pred_paths = {tags[0]: args.preds_11k, tags[1]: args.preds_14k}

    results_a = evaluate_from_predictions(args.preds_11k, tags[0], samples, args)
    results_b = evaluate_from_predictions(args.preds_14k, tags[1], samples, args)

    detailed_fields = ["id", "grade_level", "topic", "question_type", "pred_final_answer", "ref_final_answer",
                        "answer_was_unstructured"] + METRIC_KEYS

    # All output filenames carry a "(1)" suffix to keep this run's outputs
    # separate from any earlier evaluation run in the same --out-dir.
    write_csv(out_dir / f"detailed_results_{tags[0]}(1).csv", results_a, detailed_fields)
    write_csv(out_dir / f"detailed_results_{tags[1]}(1).csv", results_b, detailed_fields)

    summary_a = aggregate(results_a)
    summary_b = aggregate(results_b)

    grouped_a = {
        "grade_level": grouped_aggregate(results_a, "grade_level"),
        "topic": grouped_aggregate(results_a, "topic"),
        "question_type": grouped_aggregate(results_a, "question_type"),
    }
    grouped_b = {
        "grade_level": grouped_aggregate(results_b, "grade_level"),
        "topic": grouped_aggregate(results_b, "topic"),
        "question_type": grouped_aggregate(results_b, "question_type"),
    }

    for tag, grouped in [(tags[0], grouped_a), (tags[1], grouped_b)]:
        write_group_csv(out_dir / f"per_grade_{tag}(1).csv", grouped["grade_level"], "grade_level")
        write_group_csv(out_dir / f"per_topic_{tag}(1).csv", grouped["topic"], "topic")
        write_group_csv(out_dir / f"per_question_type_{tag}(1).csv", grouped["question_type"], "question_type")

    summary = {tags[0]: summary_a, tags[1]: summary_b, "predictions": pred_paths, "test_file": args.test_file}
    (out_dir / "summary(1).json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    metrics_rows = [dict(checkpoint=tags[0], **summary_a), dict(checkpoint=tags[1], **summary_b)]
    write_csv(out_dir / "metrics(1).csv", metrics_rows,
              ["checkpoint"] + METRIC_KEYS + ["n_samples", "n_unstructured_output"])

    write_report(out_dir / "report(1).md", summary_a, summary_b, grouped_a, grouped_b, pred_paths, tags)

    print(f"\n[done] Results written to {out_dir.resolve()}")
    print(f"  - detailed_results_{tags[0]}(1).csv / detailed_results_{tags[1]}(1).csv")
    print(f"  - per_grade_(1).csv / per_topic_(1).csv / per_question_type_*(1).csv")
    print(f"  - summary(1).json / metrics(1).csv / report(1).md")


if __name__ == "__main__":
    main()