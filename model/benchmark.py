"""
Benchmark harness for the code-comment Transformer model.

Provides:
    1. Frozen benchmark set export (Phase 0)
    2. Full evaluation suite: BLEU-4 (smoothed), ROUGE-L, token metrics,
       per-category tracking, model-vs-fallback win rate (Phase 1A)
    3. Run comparison between any two experiment reports (Phase 1A)

Usage:
    python benchmark.py --export-set          # Export frozen benchmark set
    python benchmark.py --evaluate            # Run full evaluation on frozen set
    python benchmark.py --compare A.json B.json  # Compare two runs
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from dataset import (
    BPETokenizer,
    FormattingPipe,
    SampleRecord,
    PAD_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    MAX_SEQ_LEN,
    BPE_VOCAB_SIZE,
    RANDOM_SEED,
    DEFAULT_LANGUAGES,
    DEFAULT_SOURCES,
    MAX_SAMPLES_PER_LANG,
    _collect_codesearchnet_pairs,
    _collect_codexglue_pairs,
    _collect_local_jsonl_pairs,
    _deduplicate_samples,
    _repo_aware_split,
    _normalize_code_key,
    LOCAL_JSONL_GLOB,
)
from model import TransformerConfig, TransformerDecoder
from predict import (
    DecodeConfig,
    predict_with_meta,
    _build_descriptive_fallback,
    _is_low_quality_comment,
    _normalize_spaces,
    _extract_function_name,
    _extract_intents,
    load_model,
    load_tokenizer,
)

BENCHMARK_SET_PATH = os.path.join(SCRIPT_DIR, "benchmark_set.json")
BASELINE_MANIFEST_PATH = os.path.join(SCRIPT_DIR, "baseline_manifest.json")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "benchmark_reports")


# ── Ngram / BLEU / ROUGE Utilities ───────────────────────────────────────────

def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def smoothed_sentence_bleu(ref: list[str], hyp: list[str], max_n: int = 4) -> float:
    """
    Sentence-level BLEU with add-1 smoothing (Lin & Och, 2004).
    Fixes the BLEU=0.0 bug caused by zero higher-order n-gram matches on short refs.
    """
    if len(hyp) == 0:
        return 0.0

    scores = []
    for n in range(1, max_n + 1):
        ref_ng = Counter(_ngrams(ref, n))
        hyp_ng = Counter(_ngrams(hyp, n))
        clipped = sum(min(hyp_ng[g], ref_ng[g]) for g in hyp_ng)
        total = max(sum(hyp_ng.values()), 1)
        # Add-1 smoothing for n > 1
        if n == 1:
            scores.append(clipped / total)
        else:
            scores.append((clipped + 1) / (total + 1))

    if any(s == 0 for s in scores):
        return 0.0

    log_avg = sum(math.log(s) for s in scores) / len(scores)
    bp = min(1.0, math.exp(1 - len(ref) / max(len(hyp), 1)))
    return bp * math.exp(log_avg)


def raw_sentence_bleu(ref: list[str], hyp: list[str], max_n: int = 4) -> float:
    """Unsmoothed BLEU-4 for backward compatibility."""
    if len(hyp) == 0:
        return 0.0

    scores = []
    for n in range(1, max_n + 1):
        ref_ng = Counter(_ngrams(ref, n))
        hyp_ng = Counter(_ngrams(hyp, n))
        clipped = sum(min(hyp_ng[g], ref_ng[g]) for g in hyp_ng)
        total = max(sum(hyp_ng.values()), 1)
        scores.append(clipped / total)

    if any(s == 0 for s in scores):
        return 0.0

    log_avg = sum(math.log(s) for s in scores) / len(scores)
    bp = min(1.0, math.exp(1 - len(ref) / max(len(hyp), 1)))
    return bp * math.exp(log_avg)


def rouge_l_f1(ref: list[str], hyp: list[str]) -> float:
    """ROUGE-L F1 score using longest common subsequence."""
    if not ref or not hyp:
        return 0.0

    m, n = len(ref), len(hyp)
    # LCS via DP
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / n
    recall = lcs_len / m
    return 2 * precision * recall / (precision + recall)


# ── Intent Classification ────────────────────────────────────────────────────

def classify_intent(code: str) -> str:
    """Classify code snippet into a semantic intent category."""
    intents = _extract_intents(code)
    if intents:
        return intents[0]  # Primary intent

    lower = code.lower()
    if any(k in lower for k in ["class ", "def __init__", "constructor"]):
        return "class_definition"
    if any(k in lower for k in ["import ", "require(", "from "]):
        return "import"
    if any(k in lower for k in ["return ", "yield "]):
        return "computation"

    return "other"


# ── Benchmark Set Export ─────────────────────────────────────────────────────

def export_benchmark_set():
    """
    Phase 0: Export the validation split with full metadata to a frozen JSON file.
    This ensures all future experiments evaluate on the exact same data.
    """
    print("=" * 64)
    print("  EXPORTING FROZEN BENCHMARK SET")
    print("=" * 64)

    random.seed(RANDOM_SEED)

    # Collect data using the same pipeline as training
    collected_samples: list[SampleRecord] = []
    for source in DEFAULT_SOURCES:
        for lang in DEFAULT_LANGUAGES:
            try:
                if source == "codesearchnet":
                    lang_samples = _collect_codesearchnet_pairs(lang, MAX_SAMPLES_PER_LANG)
                elif source == "codexglue":
                    lang_samples = _collect_codexglue_pairs(lang, MAX_SAMPLES_PER_LANG)
                else:
                    continue

                print(f"  Collected {len(lang_samples)} pairs for {lang} from {source}")
                collected_samples.extend(lang_samples)
            except Exception as e:
                print(f"  Failed to collect {lang} from {source}: {e}")
                continue

    if LOCAL_JSONL_GLOB:
        local_limit = MAX_SAMPLES_PER_LANG * max(len(DEFAULT_LANGUAGES), 1)
        local_samples = _collect_local_jsonl_pairs(LOCAL_JSONL_GLOB, max_samples=local_limit)
        if local_samples:
            collected_samples.extend(local_samples)

    random.Random(RANDOM_SEED).shuffle(collected_samples)
    deduped_samples, duplicates_removed = _deduplicate_samples(collected_samples)
    train_records, val_records = _repo_aware_split(deduped_samples, 0.15)

    # ── Repo leakage check (Phase 1B.1) ──────────────────────────────────
    train_code_hashes = {_normalize_code_key(s.code) for s in train_records}
    val_code_hashes = {_normalize_code_key(s.code) for s in val_records}
    leakage = train_code_hashes & val_code_hashes
    if leakage:
        print(f"\n  [WARNING] Repo leakage detected: {len(leakage)} code hashes overlap between train and val!")
    else:
        print(f"\n  [OK] Zero code-level overlap between train ({len(train_records)}) and val ({len(val_records)})")

    # Export val records with full metadata
    benchmark_entries = []
    for i, sample in enumerate(val_records):
        entry = {
            "id": i,
            "code": sample.code,
            "reference_comment": sample.comment,
            "language": sample.language,
            "repo": sample.repo,
            "path": sample.path,
            "source": sample.source,
            "intent": classify_intent(sample.code),
            "function_name": _extract_function_name(sample.code) or "",
            "code_hash": _normalize_code_key(sample.code),
            "comment_word_count": len(sample.comment.split()),
        }
        benchmark_entries.append(entry)

    # Compute deterministic hash of the entire set for integrity checking
    set_content = json.dumps(benchmark_entries, sort_keys=True, ensure_ascii=True)
    set_hash = hashlib.sha256(set_content.encode("utf-8")).hexdigest()

    benchmark_data = {
        "version": "1.0",
        "created_at": datetime.datetime.now().isoformat(),
        "set_hash_sha256": set_hash,
        "total_samples": len(benchmark_entries),
        "dataset_config": {
            "sources": list(DEFAULT_SOURCES),
            "languages": list(DEFAULT_LANGUAGES),
            "max_samples_per_lang": MAX_SAMPLES_PER_LANG,
            "val_split": 0.15,
            "random_seed": RANDOM_SEED,
            "raw_collected": len(collected_samples),
            "duplicates_removed": duplicates_removed,
            "train_samples": len(train_records),
            "val_samples": len(val_records),
            "code_leakage_count": len(leakage),
        },
        "category_counts": {},
        "language_counts": {},
        "samples": benchmark_entries,
    }

    # Compute category and language breakdowns
    for entry in benchmark_entries:
        intent = entry["intent"]
        lang = entry["language"]
        benchmark_data["category_counts"][intent] = benchmark_data["category_counts"].get(intent, 0) + 1
        benchmark_data["language_counts"][lang] = benchmark_data["language_counts"].get(lang, 0) + 1

    with open(BENCHMARK_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)

    print(f"\n  Benchmark set exported: {BENCHMARK_SET_PATH}")
    print(f"  Total samples: {len(benchmark_entries)}")
    print(f"  Set hash: {set_hash[:16]}...")
    print(f"  Categories: {json.dumps(benchmark_data['category_counts'], indent=4)}")
    print(f"  Languages: {json.dumps(benchmark_data['language_counts'], indent=4)}")

    return benchmark_data


# ── Full Evaluation Suite ────────────────────────────────────────────────────

def run_evaluation(
    checkpoint_tag: str = "current",
    decode_config: DecodeConfig | None = None,
    max_samples: int | None = None,
) -> dict:
    """
    Run full evaluation on the frozen benchmark set.

    Metrics computed:
        - Smoothed BLEU-4 (overall and per-category)
        - Raw BLEU-4
        - ROUGE-L F1 (overall and per-category)
        - Fallback trigger rate
        - Model-vs-fallback win rate (via BLEU comparison)
        - Per-language and per-category breakdowns
    """
    print("=" * 64)
    print(f"  BENCHMARK EVALUATION — {checkpoint_tag}")
    print("=" * 64)

    # Load benchmark set
    if not os.path.isfile(BENCHMARK_SET_PATH):
        print(f"  [ERROR] Benchmark set not found at {BENCHMARK_SET_PATH}")
        print(f"  Run: python benchmark.py --export-set")
        sys.exit(1)

    with open(BENCHMARK_SET_PATH, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    samples = benchmark_data["samples"]
    if max_samples is not None:
        samples = samples[:max_samples]

    print(f"  Evaluating on {len(samples)} samples")
    print(f"  Set hash: {benchmark_data['set_hash_sha256'][:16]}...")

    cfg = decode_config or DecodeConfig()

    # ── Run inference on each sample ─────────────────────────────────────
    results: list[dict] = []
    fallback_count = 0
    model_wins = 0
    total_compared = 0

    t0 = time.time()
    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            speed = (i + 1) / max(elapsed, 1e-6)
            print(f"    [{i + 1}/{len(samples)}] ({speed:.1f} samples/s)")

        code = sample["code"]
        ref_comment = sample["reference_comment"]
        ref_tokens = ref_comment.lower().split()

        # Model prediction
        model_output = predict_with_meta(code, config=cfg)
        model_comment = model_output["comment"]
        model_tokens = model_comment.lower().split()
        used_fallback = model_output["used_fallback"]

        if used_fallback:
            fallback_count += 1

        # Fallback prediction (always compute for comparison)
        fallback_comment, fallback_rule = _build_descriptive_fallback(code)
        fallback_tokens = fallback_comment.lower().split()

        # Compute metrics
        bleu_smoothed = smoothed_sentence_bleu(ref_tokens, model_tokens)
        bleu_raw = raw_sentence_bleu(ref_tokens, model_tokens)
        rouge_l = rouge_l_f1(ref_tokens, model_tokens)

        fallback_bleu = smoothed_sentence_bleu(ref_tokens, fallback_tokens)
        fallback_rouge = rouge_l_f1(ref_tokens, fallback_tokens)

        # Win rate: model vs fallback
        model_score_combined = bleu_smoothed + rouge_l
        fallback_score_combined = fallback_bleu + fallback_rouge
        model_wins_this = model_score_combined > fallback_score_combined
        if model_wins_this:
            model_wins += 1
        total_compared += 1

        result = {
            "id": sample["id"],
            "language": sample["language"],
            "intent": sample["intent"],
            "reference": ref_comment,
            "model_comment": model_comment,
            "fallback_comment": fallback_comment,
            "used_fallback": used_fallback,
            "bleu4_smoothed": round(bleu_smoothed, 6),
            "bleu4_raw": round(bleu_raw, 6),
            "rouge_l_f1": round(rouge_l, 6),
            "fallback_bleu4": round(fallback_bleu, 6),
            "fallback_rouge_l": round(fallback_rouge, 6),
            "model_beats_fallback": model_wins_this,
            "latency_ms": model_output.get("latency_ms", 0),
        }
        results.append(result)

    elapsed_total = time.time() - t0

    # ── Aggregate metrics ────────────────────────────────────────────────
    n = len(results)

    overall = {
        "bleu4_smoothed": round(sum(r["bleu4_smoothed"] for r in results) / max(n, 1), 6),
        "bleu4_raw": round(sum(r["bleu4_raw"] for r in results) / max(n, 1), 6),
        "rouge_l_f1": round(sum(r["rouge_l_f1"] for r in results) / max(n, 1), 6),
        "fallback_trigger_rate": round(fallback_count / max(n, 1), 4),
        "model_vs_fallback_win_rate": round(model_wins / max(total_compared, 1), 4),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / max(n, 1), 2),
        "total_samples": n,
        "total_time_sec": round(elapsed_total, 2),
    }

    # Per-language breakdown
    by_language: dict[str, dict] = defaultdict(lambda: {"bleu": [], "rouge": [], "wins": 0, "total": 0, "fallbacks": 0})
    for r in results:
        lang = r["language"]
        by_language[lang]["bleu"].append(r["bleu4_smoothed"])
        by_language[lang]["rouge"].append(r["rouge_l_f1"])
        by_language[lang]["total"] += 1
        if r["model_beats_fallback"]:
            by_language[lang]["wins"] += 1
        if r["used_fallback"]:
            by_language[lang]["fallbacks"] += 1

    language_metrics = {}
    for lang, data in by_language.items():
        count = data["total"]
        language_metrics[lang] = {
            "bleu4_smoothed": round(sum(data["bleu"]) / max(count, 1), 6),
            "rouge_l_f1": round(sum(data["rouge"]) / max(count, 1), 6),
            "win_rate": round(data["wins"] / max(count, 1), 4),
            "fallback_rate": round(data["fallbacks"] / max(count, 1), 4),
            "count": count,
        }

    # Per-category breakdown
    by_category: dict[str, dict] = defaultdict(lambda: {"bleu": [], "rouge": [], "wins": 0, "total": 0, "fallbacks": 0})
    for r in results:
        cat = r["intent"]
        by_category[cat]["bleu"].append(r["bleu4_smoothed"])
        by_category[cat]["rouge"].append(r["rouge_l_f1"])
        by_category[cat]["total"] += 1
        if r["model_beats_fallback"]:
            by_category[cat]["wins"] += 1
        if r["used_fallback"]:
            by_category[cat]["fallbacks"] += 1

    category_metrics = {}
    for cat, data in by_category.items():
        count = data["total"]
        category_metrics[cat] = {
            "bleu4_smoothed": round(sum(data["bleu"]) / max(count, 1), 6),
            "rouge_l_f1": round(sum(data["rouge"]) / max(count, 1), 6),
            "win_rate": round(data["wins"] / max(count, 1), 4),
            "fallback_rate": round(data["fallbacks"] / max(count, 1), 4),
            "count": count,
        }

    # ── Build report ─────────────────────────────────────────────────────
    report = {
        "experiment_tag": checkpoint_tag,
        "timestamp": datetime.datetime.now().isoformat(),
        "benchmark_set_hash": benchmark_data["set_hash_sha256"],
        "decode_config": {
            "mode": cfg.mode,
            "beam_width": cfg.beam_width,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "repetition_penalty": cfg.repetition_penalty,
            "max_len": cfg.max_len,
            "min_len": cfg.min_len,
        },
        "overall": overall,
        "by_language": language_metrics,
        "by_category": category_metrics,
        "per_sample_results": results,
    }

    # Save report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(
        REPORTS_DIR,
        f"benchmark_{checkpoint_tag}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print(f"  BENCHMARK RESULTS — {checkpoint_tag}")
    print(f"{'=' * 64}")
    print(f"  BLEU-4 (smoothed) : {overall['bleu4_smoothed']:.4f}")
    print(f"  BLEU-4 (raw)      : {overall['bleu4_raw']:.4f}")
    print(f"  ROUGE-L F1        : {overall['rouge_l_f1']:.4f}")
    print(f"  Fallback rate     : {overall['fallback_trigger_rate']:.2%}")
    print(f"  Model win rate    : {overall['model_vs_fallback_win_rate']:.2%}")
    print(f"  Avg latency       : {overall['avg_latency_ms']:.1f} ms")
    print(f"  Total time        : {overall['total_time_sec']:.1f} s")
    print(f"\n  By language:")
    for lang, m in sorted(language_metrics.items()):
        print(f"    {lang:12s}  BLEU={m['bleu4_smoothed']:.4f}  ROUGE={m['rouge_l_f1']:.4f}  win={m['win_rate']:.2%}  n={m['count']}")
    print(f"\n  By category:")
    for cat, m in sorted(category_metrics.items()):
        print(f"    {cat:20s}  BLEU={m['bleu4_smoothed']:.4f}  ROUGE={m['rouge_l_f1']:.4f}  win={m['win_rate']:.2%}  n={m['count']}")
    print(f"\n  Report saved: {report_path}")

    return report


# ── Run Comparison ───────────────────────────────────────────────────────────

def compare_runs(report_path_a: str, report_path_b: str):
    """Compare two benchmark reports side by side."""
    with open(report_path_a, "r", encoding="utf-8") as f:
        report_a = json.load(f)
    with open(report_path_b, "r", encoding="utf-8") as f:
        report_b = json.load(f)

    tag_a = report_a["experiment_tag"]
    tag_b = report_b["experiment_tag"]
    oa = report_a["overall"]
    ob = report_b["overall"]

    print(f"\n{'=' * 64}")
    print(f"  COMPARISON: {tag_a} vs {tag_b}")
    print(f"{'=' * 64}")

    metrics = ["bleu4_smoothed", "bleu4_raw", "rouge_l_f1", "model_vs_fallback_win_rate", "fallback_trigger_rate"]
    labels = ["BLEU-4 (smooth)", "BLEU-4 (raw)", "ROUGE-L F1", "Model win rate", "Fallback rate"]

    print(f"\n  {'Metric':<22s}  {tag_a:>12s}  {tag_b:>12s}  {'Δ':>10s}  {'Winner':>8s}")
    print(f"  {'─' * 22}  {'─' * 12}  {'─' * 12}  {'─' * 10}  {'─' * 8}")

    for metric, label in zip(metrics, labels):
        va = oa.get(metric, 0)
        vb = ob.get(metric, 0)
        delta = vb - va
        # For fallback rate, lower is better
        if metric == "fallback_trigger_rate":
            winner = tag_b if delta < 0 else tag_a if delta > 0 else "tie"
        else:
            winner = tag_b if delta > 0 else tag_a if delta < 0 else "tie"
        print(f"  {label:<22s}  {va:>12.4f}  {vb:>12.4f}  {delta:>+10.4f}  {winner:>8s}")

    # Per-language comparison
    all_langs = sorted(set(list(report_a.get("by_language", {}).keys()) + list(report_b.get("by_language", {}).keys())))
    if all_langs:
        print(f"\n  Per-language BLEU-4 (smoothed):")
        for lang in all_langs:
            va = report_a.get("by_language", {}).get(lang, {}).get("bleu4_smoothed", 0)
            vb = report_b.get("by_language", {}).get(lang, {}).get("bleu4_smoothed", 0)
            delta = vb - va
            print(f"    {lang:12s}  {va:.4f} → {vb:.4f}  (Δ={delta:+.4f})")

    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark harness for code-comment model")
    parser.add_argument("--export-set", action="store_true", help="Export frozen benchmark set from data sources")
    parser.add_argument("--evaluate", action="store_true", help="Run full evaluation on frozen benchmark set")
    parser.add_argument("--compare", nargs=2, metavar=("REPORT_A", "REPORT_B"), help="Compare two benchmark reports")
    parser.add_argument("--tag", default="current", help="Experiment tag for this evaluation run")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap number of samples to evaluate")
    parser.add_argument("--mode", choices=["greedy", "top_k", "top_p", "beam"], default="beam")
    parser.add_argument("--temperature", type=float, default=0.65)
    parser.add_argument("--beam-width", type=int, default=6)

    args = parser.parse_args()

    if args.export_set:
        export_benchmark_set()
    elif args.evaluate:
        cfg = DecodeConfig(
            mode=args.mode,
            temperature=args.temperature,
            beam_width=args.beam_width,
        )
        run_evaluation(
            checkpoint_tag=args.tag,
            decode_config=cfg,
            max_samples=args.max_samples,
        )
    elif args.compare:
        compare_runs(args.compare[0], args.compare[1])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
