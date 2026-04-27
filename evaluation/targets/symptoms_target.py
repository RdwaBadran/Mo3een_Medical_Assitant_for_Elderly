"""
evaluation/targets/symptoms_target.py
---------------------------------------
LangSmith target wrapper for the symptoms_analysis tool.

Signature expected by LangSmith evaluate():
    f(inputs: dict) -> dict

Wraps the real tool from:
    agent.tools.symptoms.symptoms_tool.symptoms_analysis

The actual tool signature is:
    symptoms_analysis(symptoms: str, language: str = None) -> str
    (decorated with @tool, so we call .invoke())
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def symptoms_target(inputs: dict) -> dict:
    """
    Wraps symptoms_analysis for LangSmith evaluation.

    Expected inputs keys:
        - query (str):    The symptom description text
        - language (str): Optional, 'ar' or 'en'

    Returns:
        dict with keys: output, tool_used, language, error (if any)
    """
    # Import here to avoid circular imports and module-level side effects
    from agent.tools.symptoms.symptoms_tool import symptoms_analysis

    query = inputs.get("query", "")
    language = inputs.get("language", "en")

    try:
        result = symptoms_analysis.invoke({
            "symptoms": query,
            "language": language,
        })
        return {
            "output":    result,
            "tool_used": "symptoms_analysis",
            "language":  language,
        }
    except Exception as e:
        logger.error(f"[symptoms_target] Error: {e}")
        return {
            "output":    "",
            "error":     str(e),
            "tool_used": "symptoms_analysis",
            "language":  language,
        }
