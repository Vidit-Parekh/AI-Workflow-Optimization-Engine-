"""
test_classifier_accuracy.py — Accuracy Evaluation Suite
=========================================================
Tests the classifier against 30 hand-labelled ground truth tasks
spanning 3 categories and 3 industries.

Ground truth labels are based on:
  - Published RPA/AI automation literature
  - BPI Challenge 2019 batch vs human user distinction
  - McKinsey Global Institute automation potential scores

Metrics reported
----------------
  Overall accuracy         (correct / total)
  Per-label precision      (TP / (TP + FP))
  Per-label recall         (TP / (TP + FN))
  Per-label F1 score
  Confusion matrix
  Low-confidence flags     (tasks where model is uncertain)

Run
---
  python src/test_classifier_accuracy.py

Author  : AI Workflow Optimizer Project
Cost    : $0
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
for p in [str(SRC), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from parser     import WorkflowTask
from classifier import WorkflowClassifier, AILabel

logging.basicConfig(
    level  = logging.WARNING,   # suppress INFO during test run
    format = "%(asctime)s  [%(levelname)s]  %(message)s",
)

# ---------------------------------------------------------------------------
# Ground truth test set — 30 tasks, 10 per label
# ---------------------------------------------------------------------------
# Format: (task_name, expected_label, frequency, industry)
# Labels are based on published automation research + BPI 2019 batch/human split

GROUND_TRUTH: list[tuple[str, str, int, str]] = [

    # ── AUTOMATABLE (10) ─────────────────────────────────────────────
    # Rule-based, repetitive, no judgment required
    ("Send payment confirmation email",          "automatable", 50000, "finance"),
    ("Generate monthly sales report",            "automatable", 12000, "retail"),
    ("Log transaction to database",              "automatable", 250000,"finance"),
    ("Route support ticket to correct queue",    "automatable", 30000, "IT"),
    ("Extract invoice data from PDF",            "automatable", 80000, "finance"),
    ("Schedule delivery notification",           "automatable", 45000, "logistics"),
    ("Validate purchase order fields",           "automatable", 95000, "procurement"),
    ("Archive completed case records",           "automatable", 20000, "legal"),
    ("Calculate tax on invoice",                 "automatable", 110000,"finance"),
    ("Match goods receipt to purchase order",    "automatable", 88000, "procurement"),

    # ── AUGMENTABLE (10) ─────────────────────────────────────────────
    # Requires human decision but AI can assist/recommend
    ("Review flagged fraud transaction",         "augmentable", 5000,  "finance"),
    ("Approve budget exception request",         "augmentable", 3000,  "finance"),
    ("Assess supplier risk before onboarding",   "augmentable", 800,   "procurement"),
    ("Draft response to customer complaint",     "augmentable", 15000, "customer service"),
    ("Classify insurance claim severity",        "augmentable", 25000, "insurance"),
    ("Prioritise unresolved support tickets",    "augmentable", 40000, "IT"),
    ("Verify employee expense report",           "augmentable", 18000, "HR"),
    ("Recommend product to customer",            "augmentable", 60000, "retail"),
    ("Screen job application for shortlist",     "augmentable", 9000,  "HR"),
    ("Analyse customer churn risk",              "augmentable", 35000, "telecom"),

    # ── NON-AI (10) ──────────────────────────────────────────────────
    # Requires uniquely human judgment, ethics, or relationships
    ("Negotiate vendor contract terms",          "non_ai",      500,   "procurement"),
    ("Conduct performance review interview",     "non_ai",      2000,  "HR"),
    ("Resolve executive escalation complaint",   "non_ai",      200,   "customer service"),
    ("Decide company acquisition strategy",      "non_ai",      10,    "strategy"),
    ("Mediate employee conflict",                "non_ai",      150,   "HR"),
    ("Present quarterly results to board",       "non_ai",      40,    "finance"),
    ("Build trust relationship with key client", "non_ai",      300,   "sales"),
    ("Make final hiring decision",               "non_ai",      1500,  "HR"),
    ("Lead crisis communications response",      "non_ai",      50,    "PR"),
    ("Design corporate ethics policy",           "non_ai",      5,     "strategy"),
]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    task_name    : str
    industry     : str
    expected     : str
    predicted    : str
    confidence   : float
    correct      : bool
    all_scores   : dict


def run_accuracy_test(
    ground_truth: list[tuple] = GROUND_TRUTH,
    confidence_threshold: float = 0.60,
) -> pd.DataFrame:
    """
    Run classifier on all ground truth tasks and compute accuracy metrics.

    Parameters
    ----------
    ground_truth         : list of (task_name, expected_label, freq, industry)
    confidence_threshold : tasks below this are flagged as uncertain

    Returns
    -------
    pd.DataFrame with full per-task results
    """
    print("\n" + "=" * 65)
    print("  CLASSIFIER ACCURACY TEST")
    print("  30 hand-labelled tasks  |  3 labels  |  4 industries")
    print("=" * 65)
    print("  Loading BART model (cached after first run)...")

    classifier = WorkflowClassifier(use_local=True)

    # Build WorkflowTask objects from ground truth
    tasks = [
        WorkflowTask(
            task_id   = f"test_{i+1:03d}",
            name      = name,
            source    = "test",
            frequency = freq,
        )
        for i, (name, _, freq, _) in enumerate(ground_truth)
    ]

    print(f"  Classifying {len(tasks)} tasks...\n")

    # Classify all
    results = classifier.classify_all(tasks, save_to_db=False)

    # Build result objects
    test_results: list[TestResult] = []
    for result, (name, expected, freq, industry) in zip(results, ground_truth):
        predicted = result.label.value
        correct   = predicted == expected
        test_results.append(TestResult(
            task_name  = name,
            industry   = industry,
            expected   = expected,
            predicted  = predicted,
            confidence = result.confidence,
            correct    = correct,
            all_scores = result.scores,
        ))

    return _report(test_results, confidence_threshold)


def _report(
    results             : list[TestResult],
    confidence_threshold: float,
) -> pd.DataFrame:
    """Compute and print all metrics, return DataFrame."""

    labels = [l.value for l in AILabel]
    total  = len(results)
    correct= sum(r.correct for r in results)

    # ── Per-task table ────────────────────────────────────────────────
    print(f"  {'TASK':<40} {'EXPECTED':<14} {'PREDICTED':<14} {'CONF':>5}  {'OK'}")
    print("  " + "-" * 79)
    for r in results:
        tick = "✓" if r.correct else "✗"
        flag = " ⚠ low conf" if r.confidence < confidence_threshold else ""
        print(
            f"  {r.task_name:<40} {r.expected:<14} {r.predicted:<14} "
            f"{r.confidence:>5.2f}  {tick}{flag}"
        )

    # ── Overall accuracy ──────────────────────────────────────────────
    accuracy = correct / total * 100
    print(f"\n{'=' * 65}")
    print(f"  OVERALL ACCURACY: {correct}/{total}  ({accuracy:.1f}%)")
    print(f"{'=' * 65}")

    # ── Confusion matrix ──────────────────────────────────────────────
    print("\n  CONFUSION MATRIX")
    print(f"  {'':>14}", end="")
    for l in labels:
        print(f"  {'pred_'+l:<16}", end="")
    print()
    print("  " + "-" * 65)

    for true_label in labels:
        print(f"  {'true_'+true_label:<14}", end="")
        for pred_label in labels:
            count = sum(
                1 for r in results
                if r.expected == true_label and r.predicted == pred_label
            )
            marker = " ◀" if true_label == pred_label else ""
            print(f"  {count:<16}{marker}"[: 18], end="")
        print()

    # ── Per-label metrics ─────────────────────────────────────────────
    print("\n  PER-LABEL METRICS")
    print(f"  {'LABEL':<14} {'PRECISION':>10} {'RECALL':>8} {'F1':>8} {'SUPPORT':>8}")
    print("  " + "-" * 55)

    per_label_rows = []
    for label in labels:
        tp = sum(1 for r in results if r.expected == label and r.predicted == label)
        fp = sum(1 for r in results if r.expected != label and r.predicted == label)
        fn = sum(1 for r in results if r.expected == label and r.predicted != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        support   = sum(1 for r in results if r.expected == label)

        print(
            f"  {label:<14} {precision:>10.2f} {recall:>8.2f} "
            f"{f1:>8.2f} {support:>8}"
        )
        per_label_rows.append({
            "label": label, "precision": precision,
            "recall": recall, "f1": f1, "support": support,
        })

    macro_p  = np.mean([r["precision"] for r in per_label_rows])
    macro_r  = np.mean([r["recall"]    for r in per_label_rows])
    macro_f1 = np.mean([r["f1"]        for r in per_label_rows])
    print("  " + "-" * 55)
    print(
        f"  {'macro avg':<14} {macro_p:>10.2f} {macro_r:>8.2f} "
        f"{macro_f1:>8.2f} {total:>8}"
    )

    # ── Per-industry accuracy ─────────────────────────────────────────
    print("\n  ACCURACY BY INDUSTRY")
    print(f"  {'INDUSTRY':<20} {'CORRECT':>8} {'TOTAL':>6} {'ACC':>6}")
    print("  " + "-" * 44)
    industries = sorted(set(r.industry for r in results))
    for ind in industries:
        ind_results = [r for r in results if r.industry == ind]
        ind_correct = sum(r.correct for r in ind_results)
        ind_acc     = ind_correct / len(ind_results) * 100
        print(
            f"  {ind:<20} {ind_correct:>8} {len(ind_results):>6} "
            f"{ind_acc:>5.0f}%"
        )

    # ── Low confidence flags ──────────────────────────────────────────
    low_conf = [r for r in results if r.confidence < confidence_threshold]
    if low_conf:
        print(f"\n  ⚠  LOW CONFIDENCE TASKS (< {confidence_threshold:.0%})")
        print(f"  {'TASK':<40} {'CONF':>5}  {'CORRECT?'}")
        print("  " + "-" * 55)
        for r in low_conf:
            tick = "✓" if r.correct else "✗"
            print(f"  {r.task_name:<40} {r.confidence:>5.2f}  {tick}")

    # ── Misclassified tasks ───────────────────────────────────────────
    wrong = [r for r in results if not r.correct]
    if wrong:
        print(f"\n  ✗  MISCLASSIFIED TASKS ({len(wrong)} total)")
        print(f"  {'TASK':<40} {'EXPECTED':<14} {'PREDICTED':<14} {'CONF':>5}")
        print("  " + "-" * 75)
        for r in wrong:
            print(
                f"  {r.task_name:<40} {r.expected:<14} "
                f"{r.predicted:<14} {r.confidence:>5.2f}"
            )
    else:
        print("\n  ✓  No misclassifications!")

    # ── Interpretation ────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  INTERPRETATION")
    print(f"{'=' * 65}")
    if accuracy >= 80:
        verdict = "GOOD — suitable for production use with human review"
    elif accuracy >= 65:
        verdict = "MODERATE — review low-confidence results before acting"
    else:
        verdict = "NEEDS IMPROVEMENT — consider fine-tuning or adjusting hypothesis templates"
    print(f"  Accuracy {accuracy:.1f}% → {verdict}")
    print(f"  Avg confidence: {np.mean([r.confidence for r in results]):.2f}")
    print(f"  Low-confidence tasks: {len(low_conf)}/{total}")
    print(f"  Macro F1: {macro_f1:.2f}")
    print(f"{'=' * 65}\n")

    # Return full DataFrame
    df = pd.DataFrame([{
        "task_name"  : r.task_name,
        "industry"   : r.industry,
        "expected"   : r.expected,
        "predicted"  : r.predicted,
        "confidence" : round(r.confidence, 4),
        "correct"    : r.correct,
        "score_automatable": r.all_scores.get("automatable", 0),
        "score_augmentable": r.all_scores.get("augmentable", 0),
        "score_non_ai"     : r.all_scores.get("non_ai",      0),
    } for r in results])

    return df


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = run_accuracy_test(confidence_threshold=0.60)

    # Save results to CSV
    out_path = ROOT / "data" / "accuracy_test_results.csv"
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Full results saved to: {out_path}")