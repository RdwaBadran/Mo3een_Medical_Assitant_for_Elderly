"""
evaluation/targets/agent_target.py
------------------------------------
LangSmith target wrapper for the full LangGraph ReAct agent.

The actual agent function signature is:
    run_agent(query: str, history: list | None = None) -> dict
    Returns: {"response": str, "tool_used": str | None}
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def agent_target(inputs: dict) -> dict:
    """
    Wraps run_agent for LangSmith evaluation.

    Expected inputs keys:
        - query (str):    The user query
        - language (str): Optional, for metadata only (agent detects internally)

    Returns:
        dict with keys: output, tool_used, language, error (if any)
    """
    from agent.agent import run_agent

    query = inputs.get("query", "")
    language = inputs.get("language", "en")

    try:
        result = run_agent(query)
        return {
            "output":    result.get("response", ""),
            "tool_used": result.get("tool_used"),
            "language":  language,
        }
    except Exception as e:
        logger.error(f"[agent_target] Error: {e}")
        return {
            "output":    "",
            "error":     str(e),
            "tool_used": None,
            "language":  language,
        }
