"""
agent/tools/lab/lab_tool.py
-----------------------------
The real lab_report_explanation LangChain @tool.

Full pipeline (4 steps — all free):
  1. extract_parameters()   — Groq extracts {name, value, unit} from raw text
  2. check_ranges()         — local JSON lookup, zero API calls
  3. generate_explanations()— Groq explains each parameter in patient's language
  4. report.to_markdown()   — formatted output string

Language is detected by the existing medical_guardrail.py in the agent layer.
This tool receives the language via the query context or defaults to "en".

For file uploads (PDF/image), the raw text is extracted BEFORE this tool
is called, in api/routes.py. The tool always receives plain text.
"""

from __future__ import annotations
import logging
from langchain_core.tools import tool

from agent.guardrails.medical_guardrail import detect_language
from agent.tools.lab.parameter_extractor import extract_parameters
from agent.tools.lab.range_checker import check_ranges
from agent.tools.lab.lab_llm import generate_explanations

logger = logging.getLogger(__name__)


@tool
def lab_report_explanation(report: str) -> str:
    """
    Explain the results of a medical lab report in full detail.

    Use this tool whenever the user:
    - Shares lab test values or diagnostic numbers
    - Asks what their blood test results mean
    - Provides a list of lab parameters with values
    - Mentions specific tests like HbA1c, WBC, cholesterol, creatinine, etc.

    The tool will:
    - Extract each parameter from the input
    - Compare it against the standard normal range
    - Explain what each value means in simple language
    - Flag any critical or abnormal values
    - Provide recommended next steps

    Supports Arabic and English responses automatically.

    Args:
        report: Plain text containing lab values. Can be typed by the patient
                ("my HbA1c is 7.8") or extracted from a PDF/image file.

    Returns:
        A fully formatted lab report explanation in the patient's language.
    """
    logger.info(f"[lab_tool] Received input: {report[:100]}...")

    # ── Step 0: Detect language ───────────────────────────────────────────────
    language = detect_language(report)
    logger.info(f"[lab_tool] Detected language: {language}")

    # ── Step 1: Extract parameters from raw text (Groq) ───────────────────────
    extraction = extract_parameters(report)

    if not extraction.parameters:
        if language == "ar":
            return (
                "لم أتمكن من العثور على قيم مختبرية واضحة في النص المقدم. "
                "يُرجى كتابة نتائجك بالشكل التالي:\n"
                "مثال: HbA1c: 7.8%، كرياتينين: 1.2 mg/dL، خلايا الدم البيضاء: 8.5"
            )
        return (
            "I could not find any lab values in the text you provided. "
            "Please write your results in a format like:\n"
            "Example: HbA1c: 7.8%, Creatinine: 1.2 mg/dL, WBC: 8.5 x10³/µL"
        )

    logger.info(f"[lab_tool] Extracted {len(extraction.parameters)} parameters.")

    # ── Step 2: Check against normal ranges (local, instant, free) ───────────
    checked_params = check_ranges(extraction.parameters)

    # ── Step 3: Generate explanations (Groq) ─────────────────────────────────
    lab_report = generate_explanations(checked_params, language=language)

    # ── Step 4: Format to markdown ────────────────────────────────────────────
    result = lab_report.to_markdown()
    logger.info("[lab_tool] Report formatted and ready.")
    return result