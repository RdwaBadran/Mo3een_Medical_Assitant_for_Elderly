"""
evaluation/run.py
-------------------
Single entry point for the Mo3een evaluation pipeline.

Usage:
    python -m evaluation.run
    python -m evaluation.run --experiment baseline-v1
    python -m evaluation.run --force-upload
    python -m evaluation.run --skip-worst-at-k

Orchestrates:
    1. Environment setup (LangSmith tracing, API keys)
    2. Dataset creation/upload to LangSmith
    3. Per-tool evaluation via LangSmith evaluate()
    4. Weighted aggregation (CSEDB paper formula)
    5. Worst-at-k stability analysis (optional)
    6. Report generation (terminal + JSON file)

All results are visible in LangSmith UI under project "mo3een-evaluation".
"""

from __future__ import annotations
import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment first
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluation")

# Ensure UTF-8 output on Windows consoles (emoji + Arabic support)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def setup_environment():
    """Ensure all required environment variables are set for LangSmith."""
    # Set LangSmith project for evaluation traces
    os.environ.setdefault("LANGCHAIN_PROJECT", "mo3een-evaluation")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

    # Verify required keys
    required_keys = ["GROQ_API_KEY", "LANGSMITH_API_KEY"]
    missing = [k for k in required_keys if not os.getenv(k)]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.error("Please set them in your .env file")
        sys.exit(1)

    # Also accept LANGSMITH_API_KEY as LANGCHAIN_API_KEY
    if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]

    # Also accept LANGSMITH_TRACING as LANGCHAIN_TRACING_V2
    if os.getenv("LANGSMITH_TRACING") and not os.getenv("LANGCHAIN_TRACING_V2"):
        os.environ["LANGCHAIN_TRACING_V2"] = os.environ["LANGSMITH_TRACING"]

    logger.info("✅ Environment configured for LangSmith evaluation")


def get_evaluators_for_tool(tool: str) -> list:
    """
    Return the list of evaluator functions applicable to a given tool.
    Reads from METRIC_REGISTRY to determine which evaluators apply.
    """
    from evaluation.metric_registry import METRIC_REGISTRY, get_tool_metrics
    from evaluation.evaluators import (
        # Safety
        s01_critical_illness, s02_fatal_diagnostic_error,
        s03_contraindicated_medications, s04_dose_organ_mismatch,
        s05_dose_calculation_error, s06_lethal_interaction,
        s07_antibiotic_misuse, s08_high_risk_medication_omission,
        s09_allergy_ignored, s10_suicide_risk,
        s11_medical_falsification, s12_clinical_data_errors,
        s13_polypharmacy_risk, s14_poisoning_misguidance,
        s15_public_health_misinfo, s16_vaccine_misinfo,
        s17_procedure_compliance,
        m_d01_no_dosage_advice, m_d02_doctor_referral,
        m_e01_elderly_polypharmacy,
        # Effectiveness
        e01_correct_diagnosis, e02_rare_disease_alert,
        e03_differential_coverage, e04_guideline_adherence,
        e05_multimorbidity_priority, e06_postoperative_complications,
        e07_complication_risk, e08_preventive_screening,
        e09_followup_plan, e10_lab_imaging_appropriateness,
        e11_chronic_disease_lifestyle, e12_lab_interpretation_accuracy,
        e13_combination_rationale, m_l01_arabic_quality,
        # LangSmith technical
        m_g01_refusal_accuracy, m_g02_tool_routing,
        m_g03_prompt_injection, m_g04_bias_fairness,
        # LLM Judge
        clinical_correctness_judge, elderly_language_judge,
    )

    # Map evaluator function names to actual functions
    EVALUATOR_MAP = {
        "safety_evaluators.s01_critical_illness": s01_critical_illness,
        "safety_evaluators.s02_fatal_diagnostic_error": s02_fatal_diagnostic_error,
        "safety_evaluators.s03_contraindicated_medications": s03_contraindicated_medications,
        "safety_evaluators.s04_dose_organ_mismatch": s04_dose_organ_mismatch,
        "safety_evaluators.s05_dose_calculation_error": s05_dose_calculation_error,
        "safety_evaluators.s06_lethal_interaction": s06_lethal_interaction,
        "safety_evaluators.s07_antibiotic_misuse": s07_antibiotic_misuse,
        "safety_evaluators.s08_high_risk_medication_omission": s08_high_risk_medication_omission,
        "safety_evaluators.s09_allergy_ignored": s09_allergy_ignored,
        "safety_evaluators.s10_suicide_risk": s10_suicide_risk,
        "safety_evaluators.s11_medical_falsification": s11_medical_falsification,
        "safety_evaluators.s12_clinical_data_errors": s12_clinical_data_errors,
        "safety_evaluators.s13_polypharmacy_risk": s13_polypharmacy_risk,
        "safety_evaluators.s14_poisoning_misguidance": s14_poisoning_misguidance,
        "safety_evaluators.s15_public_health_misinfo": s15_public_health_misinfo,
        "safety_evaluators.s16_vaccine_misinfo": s16_vaccine_misinfo,
        "safety_evaluators.s17_procedure_compliance": s17_procedure_compliance,
        "safety_evaluators.m_d01_no_dosage_advice": m_d01_no_dosage_advice,
        "safety_evaluators.m_d02_doctor_referral": m_d02_doctor_referral,
        "safety_evaluators.m_e01_elderly_polypharmacy": m_e01_elderly_polypharmacy,
        "effectiveness_evaluators.e01_correct_diagnosis": e01_correct_diagnosis,
        "effectiveness_evaluators.e02_rare_disease_alert": e02_rare_disease_alert,
        "effectiveness_evaluators.e03_differential_coverage": e03_differential_coverage,
        "effectiveness_evaluators.e04_guideline_adherence": e04_guideline_adherence,
        "effectiveness_evaluators.e05_multimorbidity_priority": e05_multimorbidity_priority,
        "effectiveness_evaluators.e06_postoperative_complications": e06_postoperative_complications,
        "effectiveness_evaluators.e07_complication_risk": e07_complication_risk,
        "effectiveness_evaluators.e08_preventive_screening": e08_preventive_screening,
        "effectiveness_evaluators.e09_followup_plan": e09_followup_plan,
        "effectiveness_evaluators.e10_lab_imaging_appropriateness": e10_lab_imaging_appropriateness,
        "effectiveness_evaluators.e11_chronic_disease_lifestyle": e11_chronic_disease_lifestyle,
        "effectiveness_evaluators.e12_lab_interpretation_accuracy": e12_lab_interpretation_accuracy,
        "effectiveness_evaluators.e13_combination_rationale": e13_combination_rationale,
        "effectiveness_evaluators.m_l01_arabic_quality": m_l01_arabic_quality,
        "langsmith_evaluators.m_g01_refusal_accuracy": m_g01_refusal_accuracy,
        "langsmith_evaluators.m_g02_tool_routing": m_g02_tool_routing,
        "langsmith_evaluators.m_g03_prompt_injection": m_g03_prompt_injection,
        "langsmith_evaluators.m_g04_bias_fairness": m_g04_bias_fairness,
    }

    # Get metrics for this tool
    tool_metrics = get_tool_metrics(tool)
    evaluators = set()

    for metric in tool_metrics:
        fn_name = metric.get("evaluator_fn", "")
        if fn_name in EVALUATOR_MAP:
            evaluators.add(EVALUATOR_MAP[fn_name])

    # Always add universal custom evaluators
    evaluators.add(clinical_correctness_judge)
    evaluators.add(elderly_language_judge)

    # Convert to list before adding non-hashable LangChainStringEvaluator instances
    evaluator_list = list(evaluators)

    # Add LangSmith built-in evaluators (correctness, relevance, helpfulness,
    # harmfulness, medical_groundedness) — these run via OpenAI
    from evaluation.evaluators.builtin_evaluators import BUILTIN_EVALUATORS
    evaluator_list.extend(BUILTIN_EVALUATORS)

    return evaluator_list


def run_evaluation(
    experiment: str = "baseline",
    force_upload: bool = False,
    skip_worst_at_k: bool = False,
) -> dict:
    """
    Execute the full evaluation pipeline.

    Args:
        experiment: Experiment name prefix for LangSmith
        force_upload: If True, recreate datasets even if they exist
        skip_worst_at_k: If True, skip the stability analysis

    Returns:
        Complete evaluation results dict
    """
    from langsmith import evaluate as ls_evaluate
    from evaluation.metric_registry import METRIC_REGISTRY
    from evaluation.dataset.uploader import ensure_datasets, DATASET_NAMES
    from evaluation.targets import (
        symptoms_target, drug_target, lab_target, agent_target,
    )
    from evaluation.aggregator import aggregate, format_aggregation_report

    # ── Step 1: Ensure datasets exist in LangSmith ──────────────────────────
    logger.info("Step 1/5: Ensuring datasets exist in LangSmith...")
    dataset_names = ensure_datasets(force=force_upload)
    logger.info(f"  Datasets ready: {dataset_names}")

    # ── Step 2: Define evaluation targets ────────────────────────────────────
    EVAL_TARGETS = [
        ("symptoms", symptoms_target, DATASET_NAMES["symptoms"]),
        ("drug",     drug_target,     DATASET_NAMES["drug"]),
        ("lab",      lab_target,      DATASET_NAMES["lab"]),
        ("agent",    agent_target,    DATASET_NAMES["agent"]),
    ]

    # ── Step 3: Run evaluations per tool ─────────────────────────────────────
    logger.info("Step 2/5: Running evaluations per tool...")
    all_feedback = []
    tool_results = {}

    for tool_name, target_fn, dataset_name in EVAL_TARGETS:
        logger.info(f"  Evaluating: {tool_name} → dataset={dataset_name}")
        evaluators = get_evaluators_for_tool(tool_name)
        logger.info(f"    {len(evaluators)} evaluators loaded")

        try:
            results = ls_evaluate(
                target_fn,
                data=dataset_name,
                evaluators=evaluators,
                experiment_prefix=f"{experiment}-{tool_name}",
                max_concurrency=1,
                metadata={
                    "experiment": experiment,
                    "tool": tool_name,
                    "evaluator_count": len(evaluators),
                    "framework": "CSEDB",
                    "version": "2.0",
                },
            )

            # Collect feedback from results
            # ExperimentResults is iterable, each item is an ExperimentResultRow
            # with keys: 'run', 'example', 'evaluation_results'
            tool_feedback = []
            n_cases = 0
            for row in results:
                n_cases += 1
                try:
                    eval_results = row.get("evaluation_results", None)
                    if eval_results is None:
                        continue
                    # eval_results is an EvaluationResults object with .results
                    results_list = getattr(eval_results, "results", [])
                    if not results_list and isinstance(eval_results, dict):
                        results_list = eval_results.get("results", [])
                    for er in results_list:
                        key = getattr(er, "key", None) or (er.get("key") if isinstance(er, dict) else None)
                        score = getattr(er, "score", None)
                        if score is None and isinstance(er, dict):
                            score = er.get("score")
                        if key and score is not None:
                            tool_feedback.append({
                                "key": key,
                                "score": score,
                            })
                except Exception as row_err:
                    logger.warning(f"    Failed to extract feedback from row: {row_err}")

            all_feedback.extend(tool_feedback)
            tool_results[tool_name] = {
                "n_cases": n_cases,
                "n_feedback": len(tool_feedback),
            }
            logger.info(f"    ✅ {tool_name}: {n_cases} cases, {len(tool_feedback)} scores collected")

        except Exception as e:
            logger.error(f"    ❌ {tool_name} evaluation failed: {e}")
            tool_results[tool_name] = {"error": str(e)}

    # ── Step 4: Aggregate weighted scores ────────────────────────────────────
    logger.info("Step 3/5: Aggregating weighted scores...")
    scores = aggregate(all_feedback, METRIC_REGISTRY)

    # ── Step 5: Worst-at-k stability analysis (optional) ─────────────────────
    stability = None
    if not skip_worst_at_k:
        logger.info("Step 4/5: Running Worst-at-k stability analysis...")
        try:
            from evaluation.evaluators.worst_at_k import (
                run_worst_at_k, format_worst_at_k_report,
            )
            from evaluation.dataset.generator import load_curated_cases

            # Run stability on critical cases only
            curated = load_curated_cases()
            critical_cases = []
            for tool_cases in curated.values():
                for case in tool_cases:
                    if case.get("metadata", {}).get("weight", 0) >= 5:
                        critical_cases.append(case)

            if critical_cases:
                stability = run_worst_at_k(
                    target_fn=symptoms_target,  # Primary tool for stability
                    test_cases=critical_cases[:3],  # Limit to 3 cases for speed
                    evaluator_fns=[
                        get_evaluators_for_tool("symptoms")[0],  # First evaluator
                    ],
                    n_repeats=3,
                    k_values=range(1, 4),
                )
                logger.info(f"    Stability grade: {stability['stability_grade']}")
        except Exception as e:
            logger.warning(f"    Worst-at-k analysis failed: {e}")
    else:
        logger.info("Step 4/5: Skipping Worst-at-k (--skip-worst-at-k)")

    # ── Step 6: Generate and print report ────────────────────────────────────
    logger.info("Step 5/5: Generating report...")

    report = format_aggregation_report(scores)
    print(report)

    if stability:
        from evaluation.evaluators.worst_at_k import format_worst_at_k_report
        stability_report = format_worst_at_k_report(
            stability["worst_at_k"],
            stability["stability_grade"],
        )
        print(stability_report)

    # Save JSON results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(__file__).parent / f"evaluation_results_{timestamp}.json"
    full_results = {
        "experiment": experiment,
        "timestamp": timestamp,
        "scores": scores,
        "stability": stability,
        "tool_results": tool_results,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"📄 Results saved to: {results_path}")
    logger.info(f"🔗 View in LangSmith: https://smith.langchain.com/")
    logger.info("✅ Evaluation complete!")

    return full_results


def compare_experiments(current_path: str, baseline_path: str) -> None:
    """Compare two evaluation result JSON files and print a regression report."""
    with open(current_path, encoding="utf-8") as f:
        current = json.load(f)
    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)

    current_metrics  = current["scores"].get("per_metric", {})
    baseline_metrics = baseline["scores"].get("per_metric", {})

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              EXPERIMENT COMPARISON REPORT                   ║")
    print(f"║  Baseline : {baseline.get('experiment','?'):<49}║")
    print(f"║  Current  : {current.get('experiment','?'):<49}║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  {'Metric':<35} {'Baseline':>10} {'Current':>10} {'Delta':>10}")
    print("  " + "─" * 68)

    regressions = []
    improvements = []

    all_keys = sorted(set(list(current_metrics.keys()) + list(baseline_metrics.keys())))
    for key in all_keys:
        b_score = baseline_metrics.get(key, {}).get("score")
        c_score = current_metrics.get(key, {}).get("score")
        if b_score is None or c_score is None:
            delta_str = "  N/A"
            marker = "~"
        else:
            delta = c_score - b_score
            delta_str = f"{delta:+.4f}"
            if delta < -0.05:
                marker = "❌"
                regressions.append((key, b_score, c_score, delta))
            elif delta > 0.02:
                marker = "✅"
                improvements.append((key, b_score, c_score, delta))
            else:
                marker = "  "
        b_str = f"{b_score:.4f}" if b_score is not None else "  N/A"
        c_str = f"{c_score:.4f}" if c_score is not None else "  N/A"
        print(f"  {marker} {key:<33} {b_str:>10} {c_str:>10} {delta_str:>10}")

    print()
    print(f"  Regressions : {len(regressions)}")
    print(f"  Improvements: {len(improvements)}")

    current_safety = current["scores"].get("safety_score", 0)
    baseline_safety = baseline["scores"].get("safety_score", 0)
    print()
    print(f"  Overall Safety:        {baseline_safety:.4f} → {current_safety:.4f}  ({current_safety - baseline_safety:+.4f})")
    print(f"  Overall Effectiveness: {baseline['scores'].get('effectiveness_score',0):.4f} → {current['scores'].get('effectiveness_score',0):.4f}")
    print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Mo3een Medical AI Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m evaluation.run
  python -m evaluation.run --experiment v2-safety-fix
  python -m evaluation.run --force-upload --skip-worst-at-k
  python -m evaluation.run --compare evaluation_results_20260424_024249.json
        """,
    )
    parser.add_argument(
        "--experiment",
        default="baseline",
        help="Experiment name prefix for LangSmith (default: baseline)",
    )
    parser.add_argument(
        "--force-upload",
        action="store_true",
        help="Force recreate datasets even if they already exist",
    )
    parser.add_argument(
        "--skip-worst-at-k",
        action="store_true",
        help="Skip the Worst-at-k stability analysis (faster)",
    )
    parser.add_argument(
        "--compare",
        metavar="BASELINE_JSON",
        default=None,
        help="Path to a previous results JSON file to compare against",
    )

    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         Mo3een Evaluation Pipeline — Starting               ║")
    print("║         CSEDB Clinical Safety-Effectiveness Framework       ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    setup_environment()

    results = run_evaluation(
        experiment=args.experiment,
        force_upload=args.force_upload,
        skip_worst_at_k=args.skip_worst_at_k,
    )

    # If --compare was given, print a diff against the baseline
    if args.compare:
        baseline_path = Path(args.compare)
        if not baseline_path.is_absolute():
            baseline_path = Path(__file__).parent / args.compare
        if baseline_path.exists():
            # Save current results first so compare_experiments can read it
            timestamp = results["timestamp"]
            current_path = Path(__file__).parent / f"evaluation_results_{timestamp}.json"
            compare_experiments(str(current_path), str(baseline_path))
        else:
            logger.warning(f"--compare: file not found: {baseline_path}")


if __name__ == "__main__":
    main()
