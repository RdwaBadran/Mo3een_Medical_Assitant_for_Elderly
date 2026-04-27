"""
agent/tools/drug/drug_name_extractor.py
-----------------------------------------
Uses Groq (free tier) to extract individual drug names from any natural
language input.

Input:  "Can I take Advil and warfarin together? I also take metformin."
Output: DrugNameList(drugs=["ibuprofen", "warfarin", "metformin"])

Handles:
  - Brand names (Advil → ibuprofen)
  - Multiple drugs in one sentence
  - Mixed Arabic/English drug names
  - Dosage amounts are stripped — only names returned
"""

from __future__ import annotations
import os
import json
import logging
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from agent.tools.drug.schemas.drug_schemas import DrugNameList
from agent.tools.drug.prompts.drug_prompts import build_extraction_prompt

load_dotenv()
logger = logging.getLogger(__name__)

_groq_llm: ChatGroq | None = None


def _get_groq():
    global _groq_llm
    if _groq_llm is None:
        from agent.llm_factory import get_groq_llm
        _groq_llm = get_groq_llm(model_name="llama-3.3-70b-versatile", temperature=0)
    return _groq_llm


def extract_drug_names(user_input: str) -> DrugNameList:
    """
    Extract individual drug names from a user's free-text input.

    Args:
        user_input: Raw text from the user mentioning medications.

    Returns:
        DrugNameList with a list of normalized drug names.
        Returns empty list on failure — never raises.
    """
    if not user_input or not user_input.strip():
        return DrugNameList(drugs=[], extraction_note="Empty input.")

    system_prompt, user_prompt = build_extraction_prompt(user_input.strip())

    for attempt in range(1, 3):
        try:
            llm = _get_groq()
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])

            raw = response.content.strip()
            # Strip markdown fences if model adds them
            start = raw.find('{')
            end = raw.rfind('}')
            if start == -1 or end == -1:
                raise ValueError("No JSON found in response")
            raw_json = raw[start:end + 1]

            data = json.loads(raw_json)
            result = DrugNameList(**data)
            # Clean up: lowercase, strip whitespace, remove empties
            result.drugs = [d.strip().lower() for d in result.drugs if d.strip()]
            logger.info(f"[extractor] Extracted {len(result.drugs)} drugs: {result.drugs}")
            return result

        except Exception as exc:
            logger.warning(f"[drug_extractor] Attempt {attempt} failed: {exc}")
            if attempt == 2:
                return DrugNameList(
                    drugs=[],
                    extraction_note=f"Could not extract drug names: {exc}",
                )