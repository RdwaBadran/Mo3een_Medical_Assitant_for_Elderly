"""
evaluation/targets/drug_target.py
-----------------------------------
LangSmith target wrapper for the drug_interaction_checker tool.

The actual tool signature is:
    drug_interaction_checker(drugs: str, language: str = None) -> str
    (decorated with @tool, so we call .invoke())
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def drug_target(inputs: dict) -> dict:
    """
    Wraps drug_interaction_checker for LangSmith evaluation.

    Expected inputs keys:
        - query (str):    Free text containing drug names
        - language (str): Optional, 'ar' or 'en'

    Returns:
        dict with keys: output, tool_used, language, error (if any)
    """
    from agent.tools.drug.drug_tool import drug_interaction_checker

    query = inputs.get("query", "")
    language = inputs.get("language", "en")

    try:
        result = drug_interaction_checker.invoke({
            "drugs": query,
            "language": language,
        })
        return {
            "output":    result,
            "tool_used": "drug_interaction_checker",
            "language":  language,
        }
    except Exception as e:
        logger.error(f"[drug_target] Error: {e}")
        return {
            "output":    "",
            "error":     str(e),
            "tool_used": "drug_interaction_checker",
            "language":  language,
        }
