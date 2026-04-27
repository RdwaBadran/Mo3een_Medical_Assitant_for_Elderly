"""
agent/tools/lab/parameter_extractor.py
-----------------------------------------
Uses Groq (free tier) to extract structured lab parameters
from any raw text — whether typed by the patient or parsed from a PDF.

Input:  messy raw text  e.g. "HbA1c: 7.8%, WBC 11.2 thousand, Hgb=10.5"
Output: ExtractionResult with list of RawParameter objects
"""

from __future__ import annotations
import os
import json
import logging
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import ValidationError

from agent.tools.lab.schemas.lab_schemas import RawParameter, ExtractionResult
from agent.tools.lab.prompts.lab_prompts import build_extraction_prompt

load_dotenv()
logger = logging.getLogger(__name__)

# ── Groq client singleton ─────────────────────────────────────────────────────
_groq_llm: ChatGroq | None = None


def _get_groq():
    global _groq_llm
    if _groq_llm is None:
        from agent.llm_factory import get_groq_llm
        _groq_llm = get_groq_llm(model_name="llama-3.3-70b-versatile", temperature=0)
    return _groq_llm


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_parameters(raw_text: str) -> ExtractionResult:
    """
    Use Groq to extract lab parameters from raw text.

    Args:
        raw_text: Unstructured text containing lab values.

    Returns:
        ExtractionResult with a list of RawParameter objects.
        Returns empty list on failure — never raises.
    """
    if not raw_text or not raw_text.strip():
        return ExtractionResult(parameters=[], extraction_note="Empty input text.")

    system_prompt, user_prompt = build_extraction_prompt(raw_text.strip())

    for attempt in range(1, 3):
        try:
            llm = _get_groq()
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])

            raw_output = response.content.strip()
            
            # Robust JSON extraction: Find the first '{' and last '}'
            start_idx = raw_output.find('{')
            end_idx = raw_output.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                raise json.JSONDecodeError("No JSON object found in LLM response", raw_output, 0)
            
            raw_json = raw_output[start_idx:end_idx + 1]
            data = json.loads(raw_json)
            result = ExtractionResult(**data)
            logger.info(
                f"[extractor] Attempt {attempt}: extracted "
                f"{len(result.parameters)} parameters."
            )
            return result

        except (json.JSONDecodeError, ValidationError, KeyError) as exc:
            logger.warning(f"[extractor] Attempt {attempt} parse failed: {exc}")
            if attempt == 2:
                return ExtractionResult(
                    parameters=[],
                    extraction_note=f"Could not parse lab values from the provided text: {exc}",
                )
        except Exception as exc:
            logger.error(f"[extractor] Attempt {attempt} Groq call failed: {exc}")
            if attempt == 2:
                return ExtractionResult(
                    parameters=[],
                    extraction_note="Service temporarily unavailable. Please try again.",
                )