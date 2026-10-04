#!/usr/bin/env python3
"""
evaluate.py
============================================================
Evaluates predictions produced by benchmark.py against a physics
question benchmark. This script performs evaluation ONLY:
  - It does NOT load any model.
  - It does NOT run inference.
  - It does NOT generate any answers.
  - It does NOT modify predictions.

It reads:
  - benchmark_predictions.jsonl  (model predictions)
  - physics_slm_benchmark_1000.jsonl (ground-truth benchmark)

It writes (into --output_dir):
  - evaluation_summary.json
  - evaluation_results.jsonl
  - topic_metrics.csv
  - question_type_metrics.csv
  - difficulty_metrics.csv
  - confusion_report.json
  - error_analysis.json
  - keyword_statistics.json
  - formula_statistics.json
  - evaluation_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import string
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from tabulate import tabulate
except ImportError:  # pragma: no cover
    def tabulate(rows, headers=None, tablefmt=None):
        headers = headers or []
        lines = ["\t".join(str(h) for h in headers)]
        for row in rows:
            lines.append("\t".join(str(c) for c in row))
        return "\n".join(lines)

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _SMOOTH = SmoothingFunction().method1
    _NLTK_OK = True
except ImportError:  # pragma: no cover
    _NLTK_OK = False

try:
    from rouge_score import rouge_scorer
    _ROUGE_SCORER = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    _ROUGE_OK = True
except ImportError:  # pragma: no cover
    _ROUGE_OK = False

try:
    from sklearn.metrics import precision_recall_fscore_support
    _SKLEARN_OK = True
except ImportError:  # pragma: no cover
    _SKLEARN_OK = False

try:
    import bert_score
    _BERTSCORE_OK = True
except ImportError:  # pragma: no cover
    _BERTSCORE_OK = False


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")


# ============================================================
# CONSTANTS
# ============================================================

QUESTION_CATEGORIES = [
    "Definitions",
    "Conceptual",
    "Numerical",
    "Derivations",
    "Reasoning",
    "MCQ",
    "Assertion/Reason",
    "Diagram-based",
]

DIFFICULTIES = ["Easy", "Medium", "Hard"]

ERROR_CATEGORIES = [
    "Wrong concept",
    "Wrong formula",
    "Wrong numerical value",
    "Wrong unit",
    "Missing derivation",
    "Hallucination",
    "Incomplete answer",
    "Contradictory answer",
    "Missing keywords",
    "Missing theory",
    "None",
]

CALC_PATTERN = re.compile(r"<CALC>(.*?)</CALC>", re.IGNORECASE | re.DOTALL)
NUMBER_PATTERN = re.compile(r"-?\d+\.?\d*")
UNIT_PATTERN = re.compile(
    r"\b(kg|g|m|cm|mm|km|s|ms|N|J|W|Pa|Hz|K|mol|A|V|Ω|ohm|C|T|Wb|H|F|"
    r"m/s|m/s2|m/s\^2|rad|deg|eV|kWh)\b",
    re.IGNORECASE,
)
SYMBOL_PATTERN = re.compile(r"\b[a-zA-Zα-ωΑ-Ω][a-zA-Z0-9_]*\b")
CONTRADICTION_MARKERS = [
    "however", "but", "on the other hand", "although", "not", "never",
]


# ============================================================
# LOADING
# ============================================================

def load_benchmark(path: Path) -> Tuple[Dict[str, dict], List[str]]:
    """Load the benchmark file, keyed by id. Aborts on duplicate IDs."""
    records: Dict[str, dict] = {}
    warnings: List[str] = []
    seen_ids: Counter = Counter()

    if not path.exists():
        logger.error("Benchmark file not found: %s", path)
        sys.exit(1)

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                warnings.append(f"benchmark line {line_no}: malformed JSON ({e})")
                continue

            rec_id = rec.get("id")
            if rec_id is None:
                warnings.append(f"benchmark line {line_no}: missing 'id', skipped")
                continue

            rec_id = str(rec_id)
            seen_ids[rec_id] += 1
            records[rec_id] = rec

    duplicates = [rid for rid, cnt in seen_ids.items() if cnt > 1]
    if duplicates:
        logger.error(
            "Duplicate IDs found in benchmark file (%d duplicates). Aborting. "
            "Examples: %s",
            len(duplicates), duplicates[:10],
        )
        sys.exit(1)

    for w in warnings:
        logger.warning(w)

    return records, warnings


def load_predictions(path: Path) -> Tuple[Dict[str, dict], List[str]]:
    """Load predictions file, keyed by id. Aborts on duplicate IDs."""
    records: Dict[str, dict] = {}
    warnings: List[str] = []
    seen_ids: Counter = Counter()

    if not path.exists():
        logger.error("Predictions file not found: %s", path)
        sys.exit(1)

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                warnings.append(f"predictions line {line_no}: malformed JSON ({e})")
                continue

            rec_id = rec.get("id")
            if rec_id is None:
                warnings.append(f"predictions line {line_no}: missing 'id', skipped")
                continue

            rec_id = str(rec_id)
            seen_ids[rec_id] += 1
            records[rec_id] = rec

    duplicates = [rid for rid, cnt in seen_ids.items() if cnt > 1]
    if duplicates:
        logger.error(
            "Duplicate IDs found in predictions file (%d duplicates). Aborting. "
            "Examples: %s",
            len(duplicates), duplicates[:10],
        )
        sys.exit(1)

    for w in warnings:
        logger.warning(w)

    return records, warnings


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: Optional[str], strip_punct: bool = False) -> str:
    """Lowercase, unicode-normalize, collapse whitespace, optionally strip punctuation."""
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    if strip_punct:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# METRICS
# ============================================================

def compute_exact_match(expected: str, model: str) -> int:
    return int(expected.strip() == model.strip())


def compute_normalized_exact_match(expected: str, model: str) -> int:
    a = normalize_text(expected, strip_punct=True)
    b = normalize_text(model, strip_punct=True)
    return int(a == b)


def compute_bleu(reference: str, hypothesis: str) -> float:
    ref_tokens = normalize_text(reference).split()
    hyp_tokens = normalize_text(hypothesis).split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    if not _NLTK_OK:
        # Fallback: simple n-gram overlap approximation
        ref_set, hyp_set = set(ref_tokens), set(hyp_tokens)
        overlap = len(ref_set & hyp_set)
        return overlap / max(len(hyp_set), 1)
    try:
        return float(
            sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=_SMOOTH)
        )
    except Exception:
        return 0.0


def compute_rouge(reference: str, hypothesis: str) -> Dict[str, float]:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref or not hyp:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    if not _ROUGE_OK:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    try:
        scores = _ROUGE_SCORER.score(ref, hyp)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }
    except Exception:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def compute_bertscore_batch(
    references: List[str], hypotheses: List[str]
) -> Tuple[List[float], List[float], List[float]]:
    """Batched BERTScore computation (much faster than per-record calls)."""
    n = len(references)
    if not _BERTSCORE_OK or n == 0:
        if not _BERTSCORE_OK:
            logger.warning(
                "bert-score package unavailable or model could not be loaded; "
                "BERTScore fields will be filled with 0.0"
            )
        return [0.0] * n, [0.0] * n, [0.0] * n

    safe_refs = [r if r.strip() else "empty" for r in references]
    safe_hyps = [h if h.strip() else "empty" for h in hypotheses]

    try:
        P, R, F1 = bert_score.score(
            safe_hyps,
            safe_refs,
            lang="en",
            verbose=False,
            rescale_with_baseline=False,
        )
        return P.tolist(), R.tolist(), F1.tolist()
    except Exception as e:
        logger.warning("BERTScore computation failed (%s); filling with 0.0", e)
        return [0.0] * n, [0.0] * n, [0.0] * n


def compute_keyword_match(
    expected_keywords: List[str], model_answer: str
) -> Dict[str, Any]:
    norm_answer = normalize_text(model_answer)
    keywords = [normalize_text(k) for k in (expected_keywords or []) if k]
    keywords = [k for k in keywords if k]

    if not keywords:
        return {
            "matched": [],
            "missing": [],
            "extra": [],
            "precision": None,
            "recall": None,
            "f1": None,
            "match_pct": None,
        }

    matched = [k for k in keywords if k in norm_answer]
    missing = [k for k in keywords if k not in norm_answer]

    answer_tokens = set(norm_answer.split())
    keyword_tokens = set(" ".join(keywords).split())
    extra = sorted(t for t in answer_tokens if t not in keyword_tokens)[:20]

    tp = len(matched)
    fn = len(missing)
    # "Predicted positive" approximated as matched keywords (no false-positive
    # keyword concept since keywords are a fixed reference vocabulary), so
    # precision here reflects match quality vs recall reflects coverage.
    precision = tp / max(tp, 1) if tp or fn else None
    recall = tp / len(keywords) if keywords else None

    if _SKLEARN_OK and keywords:
        y_true = [1] * len(keywords)
        y_pred = [1 if k in norm_answer else 0 for k in keywords]
        try:
            p, r, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
        except Exception:
            p, r, f1 = recall, recall, recall
    else:
        p, r = recall, recall
        f1 = (2 * p * r / (p + r)) if (p and r and (p + r) > 0) else 0.0

    return {
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "precision": float(p) if p is not None else None,
        "recall": float(r) if r is not None else None,
        "f1": float(f1) if f1 is not None else None,
        "match_pct": (tp / len(keywords)) * 100 if keywords else None,
    }


def _extract_formula_features(text: str) -> Dict[str, Any]:
    calcs = CALC_PATTERN.findall(text or "")
    has_calc = bool(calcs)
    blob = " ".join(calcs) if calcs else (text or "")
    numbers = set(NUMBER_PATTERN.findall(blob))
    units = set(u.lower() for u in UNIT_PATTERN.findall(blob))
    symbols = set(
        s for s in SYMBOL_PATTERN.findall(blob)
        if len(s) <= 3 and not s.isdigit()
    )
    return {
        "has_calc": has_calc,
        "numbers": numbers,
        "units": units,
        "symbols": symbols,
        "raw": blob,
    }


def compute_formula_match(expected_answer: str, model_answer: str) -> Dict[str, Any]:
    exp = _extract_formula_features(expected_answer)
    mod = _extract_formula_features(model_answer)

    if not exp["has_calc"]:
        return {"applicable": False, "score": None, "symbol_overlap": None,
                "both_present": None, "numbers_match": None}

    both_present = exp["has_calc"] and mod["has_calc"]

    sym_union = exp["symbols"] | mod["symbols"]
    symbol_overlap = (
        len(exp["symbols"] & mod["symbols"]) / len(sym_union) if sym_union else 0.0
    )

    num_union = exp["numbers"] | mod["numbers"]
    numbers_match = (
        len(exp["numbers"] & mod["numbers"]) / len(num_union) if num_union else 0.0
    )

    unit_union = exp["units"] | mod["units"]
    units_match = (
        len(exp["units"] & mod["units"]) / len(unit_union) if unit_union else 1.0
    )

    score = 0.0
    if both_present:
        score = 0.5 * symbol_overlap + 0.3 * numbers_match + 0.2 * units_match
    return {
        "applicable": True,
        "score": round(float(score), 4),
        "symbol_overlap": round(float(symbol_overlap), 4),
        "numbers_match": round(float(numbers_match), 4),
        "both_present": both_present,
    }


def compute_length_ratio(expected: str, model: str) -> float:
    e_len = len(normalize_text(expected).split())
    m_len = len(normalize_text(model).split())
    if e_len == 0:
        return 0.0 if m_len == 0 else float("inf")
    return round(m_len / e_len, 4)


# ============================================================
# ERROR CLASSIFICATION
# ============================================================

def classify_error(
    bench_rec: dict,
    pred_rec: dict,
    metrics: Dict[str, Any],
) -> str:
    """Heuristically classify the primary error type for a wrong answer."""
    if metrics["exact_match"] or metrics["normalized_exact_match"]:
        return "None"

    model_answer = normalize_text(pred_rec.get("model_answer", ""))
    expected_answer = normalize_text(bench_rec.get("expected_answer", ""))
    q_type = bench_rec.get("question_type", "")

    if not model_answer.strip():
        return "Incomplete answer"

    length_ratio = metrics.get("length_ratio", 1.0)
    keyword_info = metrics.get("_keyword_info", {})
    formula_info = metrics.get("_formula_info", {})

    # Formula-related errors take priority for numerical/derivation questions
    if formula_info.get("applicable"):
        if not formula_info.get("both_present"):
            return "Wrong formula"
        if formula_info.get("numbers_match", 1.0) is not None and formula_info["numbers_match"] < 0.3:
            return "Wrong numerical value"
        if formula_info.get("symbol_overlap", 1.0) < 0.3:
            return "Wrong formula"

    if q_type in ("Derivations",) and metrics.get("rougeL", 0) < 0.3:
        return "Missing derivation"

    if keyword_info.get("recall") is not None and keyword_info["recall"] < 0.4:
        return "Missing keywords"

    if length_ratio != float("inf") and length_ratio < 0.3:
        return "Incomplete answer"

    if length_ratio != float("inf") and length_ratio > 2.5:
        return "Hallucination"

    if any(marker in model_answer for marker in CONTRADICTION_MARKERS) and \
            metrics.get("rouge1", 0) < 0.5:
        return "Contradictory answer"

    if metrics.get("bertscore_f1", 0) < 0.5 and metrics.get("rouge1", 0) < 0.3:
        return "Wrong concept"

    if q_type in ("Conceptual", "Reasoning") and metrics.get("rouge1", 0) < 0.4:
        return "Missing theory"

    if UNIT_PATTERN.search(expected_answer) and not UNIT_PATTERN.search(model_answer):
        return "Wrong unit"

    return "Wrong concept"


# ============================================================
# PER-RECORD EVALUATION
# ============================================================

def evaluate_record(
    bench_rec: dict,
    pred_rec: dict,
    bertscore_triplet: Tuple[float, float, float],
) -> Dict[str, Any]:
    expected_answer = str(bench_rec.get("expected_answer", ""))
    model_answer = str(pred_rec.get("model_answer", ""))

    exact_match = compute_exact_match(expected_answer, model_answer)
    norm_exact_match = compute_normalized_exact_match(expected_answer, model_answer)
    bleu = compute_bleu(expected_answer, model_answer)
    rouge = compute_rouge(expected_answer, model_answer)
    bp, br, bf1 = bertscore_triplet
    keyword_info = compute_keyword_match(
        bench_rec.get("expected_keywords", []), model_answer
    )
    formula_info = compute_formula_match(expected_answer, model_answer)
    length_ratio = compute_length_ratio(expected_answer, model_answer)

    metrics = {
        "exact_match": exact_match,
        "normalized_exact_match": norm_exact_match,
        "bleu": round(bleu, 4),
        "rouge1": round(rouge["rouge1"], 4),
        "rouge2": round(rouge["rouge2"], 4),
        "rougeL": round(rouge["rougeL"], 4),
        "bertscore_precision": round(bp, 4),
        "bertscore_recall": round(br, 4),
        "bertscore_f1": round(bf1, 4),
        "keyword_match": (
            round(keyword_info["match_pct"], 2)
            if keyword_info["match_pct"] is not None else None
        ),
        "formula_match": formula_info["score"],
        "length_ratio": length_ratio,
        "_keyword_info": keyword_info,
        "_formula_info": formula_info,
    }

    error_category = classify_error(bench_rec, pred_rec, metrics)

    latency_ms = pred_rec.get("generation_time_ms")
    generated_tokens = pred_rec.get("generated_tokens")

    result = {
        "id": bench_rec.get("id"),
        "topic": bench_rec.get("topic", "Unknown"),
        "difficulty": bench_rec.get("difficulty", "Unknown"),
        "question_type": bench_rec.get("question_type", "Unknown"),
        "question": bench_rec.get("question", ""),
        "expected_answer": expected_answer,
        "model_answer": model_answer,
        "exact_match": exact_match,
        "normalized_exact_match": norm_exact_match,
        "bleu": metrics["bleu"],
        "rouge1": metrics["rouge1"],
        "rouge2": metrics["rouge2"],
        "rougeL": metrics["rougeL"],
        "bertscore_precision": metrics["bertscore_precision"],
        "bertscore_recall": metrics["bertscore_recall"],
        "bertscore_f1": metrics["bertscore_f1"],
        "keyword_match": metrics["keyword_match"],
        "formula_match": metrics["formula_match"],
        "length_ratio": (
            metrics["length_ratio"] if metrics["length_ratio"] != float("inf")
            else None
        ),
        "latency_ms": latency_ms,
        "generated_tokens": generated_tokens,
        "error_category": error_category,
        "_keyword_info": keyword_info,
        "_formula_info": formula_info,
        "_prompt_tokens": pred_rec.get("prompt_tokens"),
    }
    return result


# ============================================================
# AGGREGATION
# ============================================================

METRIC_COLS_FOR_CSV = [
    "exact_match", "bleu", "rougeL", "bertscore_f1", "keyword_match", "formula_match"
]


def _safe_mean(series: pd.Series) -> float:
    s = series.dropna()
    return round(float(s.mean()), 4) if len(s) else 0.0


def aggregate_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group_val, sub in df.groupby(group_col, dropna=False):
        rows.append({
            group_col: group_val,
            "Questions": len(sub),
            "Exact Match": _safe_mean(sub["exact_match"]),
            "BLEU": _safe_mean(sub["bleu"]),
            "ROUGE-L": _safe_mean(sub["rougeL"]),
            "BERTScore": _safe_mean(sub["bertscore_f1"]),
            "Keyword Match": _safe_mean(sub["keyword_match"]),
            "Formula Match": _safe_mean(sub["formula_match"]),
        })
    return pd.DataFrame(rows)


def compute_latency_stats(predictions: Dict[str, dict]) -> Dict[str, Any]:
    latencies = [
        p.get("generation_time_ms") for p in predictions.values()
        if isinstance(p.get("generation_time_ms"), (int, float))
    ]
    prompt_tokens = [
        p.get("prompt_tokens") for p in predictions.values()
        if isinstance(p.get("prompt_tokens"), (int, float))
    ]
    gen_tokens = [
        p.get("generated_tokens") for p in predictions.values()
        if isinstance(p.get("generated_tokens"), (int, float))
    ]

    def stats(values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"average": None, "median": None, "p95": None,
                    "minimum": None, "maximum": None}
        arr = np.array(values, dtype=float)
        return {
            "average": round(float(np.mean(arr)), 2),
            "median": round(float(np.median(arr)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "minimum": round(float(np.min(arr)), 2),
            "maximum": round(float(np.max(arr)), 2),
        }

    return {
        "generation_time_ms": stats(latencies),
        "prompt_tokens": stats(prompt_tokens),
        "generated_tokens": stats(gen_tokens),
    }


# ============================================================
# REPORT WRITING
# ============================================================

def write_reports(
    df: pd.DataFrame,
    predictions: Dict[str, dict],
    benchmark: Dict[str, dict],
    output_dir: Path,
    load_warnings: List[str],
    skipped_records: List[Dict[str, str]],
    elapsed_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- evaluation_results.jsonl ----
    results_path = output_dir / "evaluation_results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            clean = {k: v for k, v in rec.items() if not k.startswith("_")}
            # Replace NaN with None for valid JSON
            for k, v in clean.items():
                if isinstance(v, float) and np.isnan(v):
                    clean[k] = None
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    # ---- topic / question_type / difficulty CSVs ----
    topic_df = aggregate_metrics(df, "topic").rename(columns={"topic": "Topic"})
    topic_df.to_csv(output_dir / "topic_metrics.csv", index=False)

    qtype_df = aggregate_metrics(df, "question_type").rename(
        columns={"question_type": "Question Type"}
    )
    qtype_df.to_csv(output_dir / "question_type_metrics.csv", index=False)

    diff_df = aggregate_metrics(df, "difficulty").rename(
        columns={"difficulty": "Difficulty"}
    )
    # enforce canonical ordering where possible
    diff_df["_order"] = diff_df["Difficulty"].apply(
        lambda d: DIFFICULTIES.index(d) if d in DIFFICULTIES else 99
    )
    diff_df = diff_df.sort_values("_order").drop(columns="_order")
    diff_df.to_csv(output_dir / "difficulty_metrics.csv", index=False)

    # ---- error_analysis.json ----
    error_counts = df["error_category"].value_counts().to_dict()
    for cat in ERROR_CATEGORIES:
        error_counts.setdefault(cat, 0)
    error_analysis = {
        "counts": error_counts,
        "total_errors": int(sum(v for k, v in error_counts.items() if k != "None")),
        "error_rate": round(
            float((df["error_category"] != "None").mean()), 4
        ) if len(df) else 0.0,
    }
    with (output_dir / "error_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2, ensure_ascii=False)

    # ---- confusion_report.json (question_type x error_category) ----
    confusion = (
        df.groupby(["question_type", "error_category"])
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )
    with (output_dir / "confusion_report.json").open("w", encoding="utf-8") as f:
        json.dump(confusion, f, indent=2, ensure_ascii=False, default=int)

    # ---- keyword_statistics.json ----
    all_matched, all_missing, all_extra = [], [], []
    keyword_recalls = []
    for rec in df.to_dict(orient="records"):
        ki = rec.get("_keyword_info", {})
        if not ki:
            continue
        all_matched.extend(ki.get("matched", []))
        all_missing.extend(ki.get("missing", []))
        all_extra.extend(ki.get("extra", []))
        if ki.get("recall") is not None:
            keyword_recalls.append(ki["recall"])
    keyword_stats = {
        "average_keyword_recall": round(float(np.mean(keyword_recalls)), 4)
        if keyword_recalls else None,
        "most_frequently_missing_keywords": Counter(all_missing).most_common(20),
        "most_frequently_matched_keywords": Counter(all_matched).most_common(20),
        "most_common_extra_terms": Counter(all_extra).most_common(20),
    }
    with (output_dir / "keyword_statistics.json").open("w", encoding="utf-8") as f:
        json.dump(keyword_stats, f, indent=2, ensure_ascii=False)

    # ---- formula_statistics.json ----
    formula_applicable = df["formula_match"].notna()
    formula_stats = {
        "questions_with_formula": int(formula_applicable.sum()),
        "average_formula_match_score": (
            round(float(df.loc[formula_applicable, "formula_match"].mean()), 4)
            if formula_applicable.any() else None
        ),
        "both_present_rate": None,
    }
    both_present_flags = [
        rec.get("_formula_info", {}).get("both_present")
        for rec in df.to_dict(orient="records")
        if rec.get("_formula_info", {}).get("applicable")
    ]
    if both_present_flags:
        formula_stats["both_present_rate"] = round(
            sum(1 for b in both_present_flags if b) / len(both_present_flags), 4
        )
    with (output_dir / "formula_statistics.json").open("w", encoding="utf-8") as f:
        json.dump(formula_stats, f, indent=2, ensure_ascii=False)

    # ---- latency / overall metrics for summary ----
    latency_stats = compute_latency_stats(predictions)

    overall = {
        "exact_match": _safe_mean(df["exact_match"]),
        "normalized_exact_match": _safe_mean(df["normalized_exact_match"]),
        "bleu": _safe_mean(df["bleu"]),
        "rouge1": _safe_mean(df["rouge1"]),
        "rouge2": _safe_mean(df["rouge2"]),
        "rougeL": _safe_mean(df["rougeL"]),
        "bertscore_precision": _safe_mean(df["bertscore_precision"]),
        "bertscore_recall": _safe_mean(df["bertscore_recall"]),
        "bertscore_f1": _safe_mean(df["bertscore_f1"]),
        "keyword_match": _safe_mean(df["keyword_match"]),
        "formula_match": _safe_mean(df["formula_match"]),
        "average_generated_tokens": latency_stats["generated_tokens"]["average"],
    }

    # ---- evaluation_summary.json ----
    summary = {
        "timestamp": pd.Timestamp.now("UTC").isoformat(),
        "dataset_statistics": {
            "benchmark_questions": len(benchmark),
            "predictions_received": len(predictions),
            "questions_evaluated": len(df),
            "skipped_records": len(skipped_records),
            "load_warnings": load_warnings,
        },
        "overall_metrics": overall,
        "latency": latency_stats["generation_time_ms"],
        "token_statistics": {
            "prompt_tokens": latency_stats["prompt_tokens"],
            "generated_tokens": latency_stats["generated_tokens"],
        },
        "error_statistics": error_analysis,
        "evaluation_time_seconds": round(elapsed_seconds, 2),
    }
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- evaluation_report.md ----
    write_markdown_report(
        df, topic_df, qtype_df, diff_df, overall, latency_stats,
        error_analysis, output_dir, len(benchmark), len(predictions),
        skipped_records,
    )


def write_markdown_report(
    df: pd.DataFrame,
    topic_df: pd.DataFrame,
    qtype_df: pd.DataFrame,
    diff_df: pd.DataFrame,
    overall: Dict[str, Any],
    latency_stats: Dict[str, Any],
    error_analysis: Dict[str, Any],
    output_dir: Path,
    n_benchmark: int,
    n_predictions: int,
    skipped_records: List[Dict[str, str]],
) -> None:
    lines = []
    lines.append("# Evaluation Report\n")
    lines.append(f"_Generated: {pd.Timestamp.now('UTC').isoformat()}_\n")

    lines.append("## Dataset Overview\n")
    lines.append(f"- Benchmark questions: **{n_benchmark}**")
    lines.append(f"- Predictions received: **{n_predictions}**")
    lines.append(f"- Questions evaluated: **{len(df)}**")
    lines.append(f"- Records skipped (malformed): **{len(skipped_records)}**\n")

    lines.append("## Overall Metrics\n")
    overall_rows = [[k, v] for k, v in overall.items()]
    lines.append(tabulate(overall_rows, headers=["Metric", "Value"], tablefmt="pipe"))
    lines.append("")

    lines.append("## Topic-wise Metrics\n")
    lines.append(tabulate(topic_df, headers="keys", tablefmt="pipe", showindex=False))
    lines.append("")

    lines.append("## Question-Type Metrics\n")
    lines.append(tabulate(qtype_df, headers="keys", tablefmt="pipe", showindex=False))
    lines.append("")

    lines.append("## Difficulty Metrics\n")
    lines.append(tabulate(diff_df, headers="keys", tablefmt="pipe", showindex=False))
    lines.append("")

    lines.append("## Latency\n")
    lat_rows = [[k, v] for k, v in latency_stats["generation_time_ms"].items()]
    lines.append(tabulate(lat_rows, headers=["Stat (ms)", "Value"], tablefmt="pipe"))
    lines.append("")

    lines.append("## Error Statistics\n")
    err_rows = sorted(error_analysis["counts"].items(), key=lambda x: -x[1])
    lines.append(tabulate(err_rows, headers=["Error Category", "Count"], tablefmt="pipe"))
    lines.append(f"\nOverall error rate: **{error_analysis['error_rate'] * 100:.2f}%**\n")

    # Top 20 best / worst by a composite score (rougeL + bertscore_f1)
    scored = df.copy()
    scored["_composite"] = scored[["rougeL", "bertscore_f1"]].mean(axis=1, skipna=True)
    best = scored.sort_values("_composite", ascending=False).head(20)
    worst = scored.sort_values("_composite", ascending=True).head(20)

    lines.append("## Top 20 Best Answers\n")
    for _, row in best.iterrows():
        lines.append(
            f"- **{row['id']}** ({row['topic']}, {row['question_type']}) — "
            f"composite: {row['_composite']:.3f}, exact_match: {row['exact_match']}"
        )
    lines.append("")

    lines.append("## Top 20 Worst Answers\n")
    for _, row in worst.iterrows():
        lines.append(
            f"- **{row['id']}** ({row['topic']}, {row['question_type']}) — "
            f"composite: {row['_composite']:.3f}, error: {row['error_category']}"
        )
    lines.append("")

    lines.append("## Observations\n")
    top_error = err_rows[0][0] if err_rows and err_rows[0][0] != "None" else (
        err_rows[1][0] if len(err_rows) > 1 else "N/A"
    )
    lines.append(
        f"- The most common error category (excluding correct answers) is "
        f"**{top_error}**."
    )
    if len(topic_df):
        best_topic = topic_df.loc[topic_df["Exact Match"].idxmax()]
        worst_topic = topic_df.loc[topic_df["Exact Match"].idxmin()]
        lines.append(
            f"- Strongest topic by exact match: **{best_topic['Topic']}** "
            f"({best_topic['Exact Match']:.2f})."
        )
        lines.append(
            f"- Weakest topic by exact match: **{worst_topic['Topic']}** "
            f"({worst_topic['Exact Match']:.2f})."
        )
    lines.append("")

    lines.append("## Strengths\n")
    lines.append(
        f"- Overall normalized exact match of "
        f"{overall['normalized_exact_match'] * 100:.1f}% "
        f"and ROUGE-L of {overall['rougeL']:.3f} indicate baseline competence "
        f"on well-represented topics/question types."
    )
    lines.append("")

    lines.append("## Weaknesses\n")
    lines.append(
        f"- Formula match average of {overall['formula_match']:.3f} suggests "
        f"numerical/derivation questions may need targeted improvement."
    )
    lines.append(
        f"- Keyword match average of {overall['keyword_match']:.2f}% indicates "
        f"potential gaps in coverage of expected concepts."
    )
    lines.append("")

    lines.append("## Recommendations\n")
    lines.append(
        "- Prioritize fine-tuning or prompting improvements for the weakest "
        "topic and question-type categories identified above."
    )
    lines.append(
        "- Investigate records classified under the dominant error category "
        "for targeted data augmentation or instruction improvements."
    )
    lines.append(
        "- Re-run this evaluation after each model iteration to track "
        "regression/improvement across topics, difficulty, and question types."
    )
    lines.append("")

    (output_dir / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate benchmark.py predictions against a physics benchmark. "
                    "Evaluation only — does not load a model or run inference."
    )
    parser.add_argument(
        "--benchmark_file", type=str, default="physics_slm_benchmark_1000.jsonl"
    )
    parser.add_argument(
        "--predictions_file", type=str, default="benchmark_predictions.jsonl"
    )
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    args = parser.parse_args()

    start_time = time.time()

    benchmark_path = Path(args.benchmark_file)
    predictions_path = Path(args.predictions_file)
    output_dir = Path(args.output_dir)

    logger.info("Loading benchmark file: %s", benchmark_path)
    benchmark, bench_warnings = load_benchmark(benchmark_path)
    logger.info("Loaded %d benchmark questions", len(benchmark))

    logger.info("Loading predictions file: %s", predictions_path)
    predictions, pred_warnings = load_predictions(predictions_path)
    logger.info("Loaded %d predictions", len(predictions))

    load_warnings = bench_warnings + pred_warnings

    if len(benchmark) != len(predictions):
        logger.warning(
            "Mismatch: %d benchmark questions vs %d predictions",
            len(benchmark), len(predictions),
        )

    bench_ids = set(benchmark.keys())
    pred_ids = set(predictions.keys())
    missing_preds = bench_ids - pred_ids
    extra_preds = pred_ids - bench_ids
    if missing_preds:
        logger.warning(
            "%d benchmark IDs have no matching prediction (e.g. %s)",
            len(missing_preds), list(missing_preds)[:5],
        )
    if extra_preds:
        logger.warning(
            "%d predictions have no matching benchmark ID (e.g. %s)",
            len(extra_preds), list(extra_preds)[:5],
        )

    common_ids = sorted(bench_ids & pred_ids)
    logger.info("Evaluating %d matched records...", len(common_ids))

    skipped_records: List[Dict[str, str]] = []
    valid_pairs: List[Tuple[dict, dict]] = []
    for rid in common_ids:
        bench_rec = benchmark[rid]
        pred_rec = predictions[rid]
        try:
            if not isinstance(bench_rec.get("expected_answer", ""), (str, type(None))):
                raise ValueError("expected_answer is not text")
            if not isinstance(pred_rec.get("model_answer", ""), (str, type(None))):
                raise ValueError("model_answer is not text")
            valid_pairs.append((bench_rec, pred_rec))
        except Exception as e:
            skipped_records.append({"id": rid, "reason": str(e)})

    # Batch BERTScore computation (loads model once)
    logger.info("Computing BERTScore in batch for %d records...", len(valid_pairs))
    references = [str(b.get("expected_answer", "")) for b, _ in valid_pairs]
    hypotheses = [str(p.get("model_answer", "")) for _, p in valid_pairs]
    bp_list, br_list, bf1_list = compute_bertscore_batch(references, hypotheses)

    results = []
    for (bench_rec, pred_rec), bp, br, bf1 in tqdm(
        list(zip(valid_pairs, bp_list, br_list, bf1_list)),
        desc="Evaluating records",
    ):
        try:
            res = evaluate_record(bench_rec, pred_rec, (bp, br, bf1))
            results.append(res)
        except Exception as e:
            skipped_records.append({"id": bench_rec.get("id"), "reason": str(e)})

    if not results:
        logger.error("No records were successfully evaluated. Aborting.")
        sys.exit(1)

    df = pd.DataFrame(results)

    elapsed = time.time() - start_time
    logger.info("Evaluation complete: %d records evaluated, %d skipped",
                len(df), len(skipped_records))
    logger.info("Writing reports to %s", output_dir)

    write_reports(
        df, predictions, benchmark, output_dir, load_warnings,
        skipped_records, elapsed,
    )

    if skipped_records:
        skipped_path = output_dir / "skipped_records.json"
        with skipped_path.open("w", encoding="utf-8") as f:
            json.dump(skipped_records, f, indent=2, ensure_ascii=False)
        logger.info("Skipped-record details written to %s", skipped_path)

    total_time = time.time() - start_time
    logger.info("Done. Total time: %.2f seconds", total_time)
    logger.info(
        "Summary — questions evaluated: %d | exact match: %.3f | "
        "normalized exact match: %.3f | BLEU: %.3f | ROUGE-L: %.3f | "
        "BERTScore F1: %.3f",
        len(df),
        df["exact_match"].mean(),
        df["normalized_exact_match"].mean(),
        df["bleu"].mean(),
        df["rougeL"].mean(),
        df["bertscore_f1"].mean(),
    )


if __name__ == "__main__":
    main()