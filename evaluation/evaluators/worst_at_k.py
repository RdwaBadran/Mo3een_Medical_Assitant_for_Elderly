"""
evaluation/evaluators/worst_at_k.py
--------------------------------------
Worst-at-k stability analysis (Paper Equation 4).

Measures how stable Mo3een's performance is across repeated runs
of high-stakes test cases. A stable system should produce consistent
results; an unstable one may pass sometimes and fail others.

Formula:
    Worst@k = (1/M) × Σⱼ [ min over k samples from {s₁..sₙ} for case j ]

Where M = number of test cases, k = sample size, sᵢ = score for run i.

This module re-runs critical (weight ≥ 4) test cases multiple times
and computes the worst-case score at each k value.
"""

from __future__ import annotations
import logging
from typing import Callable

logger = logging.getLogger(__name__)


def compute_worst_at_k(
    scores_per_case: dict[str, list[float]],
    k_values: list[int] | range = range(1, 11),
) -> dict[int, float]:
    """
    Compute Worst@k for a set of repeated evaluation scores.

    Args:
        scores_per_case: Dict mapping case_id -> list of scores from repeated runs.
                         e.g. {"case_1": [1.0, 0.0, 1.0, 1.0, ...], ...}
        k_values: Values of k to compute Worst@k for.

    Returns:
        Dict mapping k -> Worst@k score (float in [0, 1]).
    """
    if not scores_per_case:
        return {k: 0.0 for k in k_values}

    results = {}
    M = len(scores_per_case)

    for k in k_values:
        total = 0.0
        for case_id, scores in scores_per_case.items():
            if len(scores) >= k:
                # Take the minimum of the first k scores
                worst = min(scores[:k])
            elif scores:
                # If fewer than k runs available, use all
                worst = min(scores)
            else:
                worst = 0.0
            total += worst

        results[k] = round(total / M, 4) if M > 0 else 0.0

    return results


def run_worst_at_k(
    target_fn: Callable,
    test_cases: list[dict],
    evaluator_fns: list[Callable],
    n_repeats: int = 5,
    k_values: list[int] | range = range(1, 6),
) -> dict:
    """
    Run worst-at-k stability analysis by re-running critical test cases.

    Args:
        target_fn: The target function to call (e.g., symptoms_target)
        test_cases: List of test case dicts with 'inputs' and 'outputs'
        evaluator_fns: List of evaluator functions to score each run
        n_repeats: Number of times to repeat each test case
        k_values: Range of k values to compute

    Returns:
        Dict with:
            - worst_at_k: {k: score} mapping
            - per_case_scores: raw scores per case
            - stability_grade: "stable" | "moderate" | "unstable"
    """
    logger.info(
        f"[worst_at_k] Running stability analysis: "
        f"{len(test_cases)} cases × {n_repeats} repeats"
    )

    scores_per_case: dict[str, list[float]] = {}

    for i, case in enumerate(test_cases):
        case_id = f"case_{i}"
        case_scores = []

        for repeat in range(n_repeats):
            try:
                # Run the target
                result = target_fn(case["inputs"])

                # Create a minimal Run-like and Example-like object for evaluators
                run_mock = _MockRun(
                    inputs=case["inputs"],
                    outputs=result,
                )
                example_mock = _MockExample(
                    inputs=case["inputs"],
                    outputs=case.get("outputs", {}),
                )

                # Average all evaluator scores for this run
                eval_scores = []
                for eval_fn in evaluator_fns:
                    eval_result = eval_fn(run_mock, example_mock)
                    if eval_result.get("score") is not None:
                        eval_scores.append(eval_result["score"])

                if eval_scores:
                    avg_score = sum(eval_scores) / len(eval_scores)
                else:
                    avg_score = 0.0

                case_scores.append(avg_score)

            except Exception as e:
                logger.warning(f"[worst_at_k] Case {case_id} repeat {repeat} failed: {e}")
                case_scores.append(0.0)

        scores_per_case[case_id] = case_scores

    # Compute worst-at-k
    worst_at_k = compute_worst_at_k(scores_per_case, k_values)

    # Determine stability grade
    if worst_at_k:
        k_max = max(k_values)
        k_min = min(k_values)
        drop = worst_at_k.get(k_min, 0) - worst_at_k.get(k_max, 0)
        if drop < 0.1:
            grade = "stable"
        elif drop < 0.25:
            grade = "moderate"
        else:
            grade = "unstable"
    else:
        grade = "unknown"

    return {
        "worst_at_k": worst_at_k,
        "per_case_scores": scores_per_case,
        "stability_grade": grade,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MOCK OBJECTS for running evaluators outside LangSmith context
# ══════════════════════════════════════════════════════════════════════════════

class _MockRun:
    """Minimal mock of langsmith.schemas.Run for evaluator compatibility."""
    def __init__(self, inputs: dict, outputs: dict):
        self.inputs = inputs
        self.outputs = outputs


class _MockExample:
    """Minimal mock of langsmith.schemas.Example for evaluator compatibility."""
    def __init__(self, inputs: dict, outputs: dict):
        self.inputs = inputs
        self.outputs = outputs


def format_worst_at_k_report(worst_at_k: dict, stability_grade: str) -> str:
    """Format worst-at-k results as a readable string."""
    lines = ["", "═══ Worst-at-k Stability Analysis ═══"]
    lines.append(f"  Stability Grade: {stability_grade.upper()}")
    lines.append("")

    for k, score in sorted(worst_at_k.items()):
        bar = "█" * int(score * 20)
        lines.append(f"  k={k:2d}  {score:.4f}  {bar}")

    lines.append("")
    return "\n".join(lines)
