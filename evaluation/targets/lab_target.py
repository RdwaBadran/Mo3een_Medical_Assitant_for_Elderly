"""
evaluation/targets/lab_target.py
----------------------------------
LangSmith target wrapper for the lab_report_explanation tool.

The actual tool signature is:
    lab_report_explanation(report: str, language: str = None) -> str
    (decorated with @tool, so we call .invoke())
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def lab_target(inputs: dict) -> dict:
    """
    Wraps lab_report_explanation for LangSmith evaluation.

    Expected inputs keys:
        - query (str):    Plain text containing lab values
        - language (str): Optional, 'ar' or 'en'

    Returns:
        dict with keys: output, tool_used, language, error (if any)
    """
    from agent.tools.lab.lab_tool import lab_report_explanation

    query = inputs.get("query", "")
    language = inputs.get("language", "en")

    try:
        result = lab_report_explanation.invoke({
            "report": query,
            "language": language,
        })
        return {
            "output":    result,
            "tool_used": "lab_report_explanation",
            "language":  language,
        }
    except Exception as e:
        logger.error(f"[lab_target] Error: {e}")
        return {
            "output":    "",
            "error":     str(e),
            "tool_used": "lab_report_explanation",
            "language":  language,
        }
