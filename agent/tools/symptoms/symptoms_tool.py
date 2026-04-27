"""
agent/tools/symptoms/symptoms_tool.py
---------------------------------------
The real symptoms_analysis LangChain tool.

Full pipeline:
  1. GuardrailResult  — language detection + medical relevance check (Groq, fast)
  2. Input validation — SymptomsInput Pydantic schema
  3. RAG retrieval    — ChromaDB similarity search (local, free)
  4. LLM diagnosis    — GPT-4o structured JSON output (OpenAI)
  5. Output schema    — SymptomsAnalysisOutput Pydantic validation
  6. Formatted text   — language-aware human-readable response

If the guardrail rejects the query, the pipeline short-circuits and
returns a polite refusal in the user's language — no OpenAI tokens spent.
"""

from __future__ import annotations
import logging
from pydantic import ValidationError
from langchain_core.tools import tool

from agent.guardrails.medical_guardrail import run_guardrails
from agent.tools.symptoms.schemas.symptom_schemas import SymptomsInput
from agent.tools.symptoms.rag import retrieve_context
from agent.tools.symptoms.llm import generate_diagnosis

logger = logging.getLogger(__name__)


@tool
def symptoms_analysis(symptoms: str, language: str = None) -> str:
    """
    Analyze a list of medical symptoms and return a diagnostic assessment.

    Use this tool whenever the user describes symptoms, pain, or discomfort.

    The tool:
    - Retreives medical knowledge from a local database
    - Generates a structured diagnosis based on that knowledge

    Args:
        symptoms: A plain-text description of the symptoms.
        language: (Optional) Response language ('ar' or 'en'). 
                 If provided, overrides detection.
    """
    logger.info(f"[symptoms_tool] Received query: {symptoms[:80]}... lang={language}")

    # ── Step 1: Guardrails & Language ────────────────────────────────────────
    if not language:
        guard = run_guardrails(symptoms)
        if not guard.passed:
            return guard.refusal_message
        language = guard.language
    else:
        # Check relevance even if language is provided
        from agent.guardrails.medical_guardrail import is_medical_query, get_refusal_message
        if not is_medical_query(symptoms):
            return get_refusal_message(language)

    # ── Step 2: Validate input schema ────────────────────────────────────────
    try:
        validated_input = SymptomsInput(
            symptoms=symptoms,
            detected_language=language,
        )
    except ValidationError as exc:
        logger.warning(f"[symptoms_tool] Input validation failed: {exc}")
        if language == "ar":
            return "يُرجى تقديم وصف أكثر تفصيلاً للأعراض التي تعاني منها."
        return "Please provide a more detailed description of your symptoms."

    # ── Step 3: RAG retrieval ─────────────────────────────────────────────────
    context = retrieve_context(validated_input.symptoms)
    if context:
        logger.info(f"[symptoms_tool] Retrieved context: {len(context)} chars")
    else:
        logger.warning("[symptoms_tool] No RAG context retrieved — LLM will use base knowledge.")

    # ── Step 4: Generate diagnosis via GPT-4o ─────────────────────────────────
    diagnosis_output = generate_diagnosis(
        symptoms=validated_input.symptoms,
        context=context,
        language=language,
    )

    # ── Step 5: Convert to readable text in the correct language ──────────────
    formatted_response = diagnosis_output.to_readable_text(language=language)
    logger.info("[symptoms_tool] Diagnosis generated and formatted successfully.")

    return formatted_response