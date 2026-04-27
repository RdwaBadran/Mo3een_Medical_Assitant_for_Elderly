"""
agent/tools/symptoms/llm.py
----------------------------
OpenAI GPT-4o caller for the symptoms diagnosis pipeline.

Responsibilities:
  1. Build the full prompt using prompts/diagnosis_prompt.py
  2. Call GPT-4o with structured JSON output mode
  3. Parse and validate the response against SymptomsAnalysisOutput schema
  4. Return a validated SymptomsAnalysisOutput object (never raw JSON)

If parsing fails (malformed JSON from the LLM), it retries once.
If the retry fails, it returns a safe fallback response.
"""

from __future__ import annotations
import os
import json
import logging
from dotenv import load_dotenv

from openai import OpenAI
from pydantic import ValidationError

from agent.tools.symptoms.schemas.symptom_schemas import SymptomsAnalysisOutput
from agent.tools.symptoms.prompts.diagnosis_prompt import (
    build_system_prompt,
    build_user_prompt,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ── OpenAI client (singleton) ─────────────────────────────────────────────────
_openai_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in .env")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


# ── Fallback response (returned if all LLM attempts fail) ────────────────────
def _fallback_response(language: str) -> SymptomsAnalysisOutput:
    """Return a safe, honest fallback when the LLM call fails."""
    if language == "ar":
        return SymptomsAnalysisOutput(
            possible_conditions=[{
                "name": "غير محدد",
                "rationale": "حدث خطأ أثناء التحليل. يُرجى المحاولة مرة أخرى أو استشارة طبيب.",
                "urgency": "medium",
            }],
            red_flags=[],
            recommended_next_steps=["يُرجى استشارة طبيب متخصص."],
            confidence="low",
            disclaimer=(
                "لم يتمكن النظام من إكمال التحليل. يُرجى التواصل مع طبيب مؤهل."
            ),
        )
    return SymptomsAnalysisOutput(
        possible_conditions=[{
            "name": "Unable to determine",
            "rationale": "An error occurred during analysis. Please try again or consult a doctor.",
            "urgency": "medium",
        }],
        red_flags=[],
        recommended_next_steps=["Please consult a qualified healthcare professional."],
        confidence="low",
        disclaimer=(
            "The system could not complete the analysis. Please contact a healthcare provider."
        ),
    )


# ── Core LLM call ─────────────────────────────────────────────────────────────
def _call_openai(system_prompt: str, user_prompt: str) -> str:
    """
    Make one call to GPT-4o with JSON response format enforced.
    Returns the raw response string.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},   # enforces valid JSON output
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.1,    # near-deterministic for clinical consistency
        max_tokens=1000,
    )
    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC FUNCTION — called by symptoms_tool.py
# ══════════════════════════════════════════════════════════════════════════════

def generate_diagnosis(
    symptoms: str,
    context: str,
    language: str = "en",
) -> SymptomsAnalysisOutput:
    """
    Call GPT-4o to generate a structured diagnosis.
    Validates the response against SymptomsAnalysisOutput schema.
    Retries once on parse failure. Returns fallback on second failure.

    Args:
        symptoms: The patient's symptom description.
        context:  Retrieved RAG context string (may be empty).
        language: ISO 639-1 language code for response language.

    Returns:
        Validated SymptomsAnalysisOutput object.
    """
    system_prompt = build_system_prompt(language)
    user_prompt = build_user_prompt(symptoms, context)

    for attempt in range(1, 3):   # try twice
        try:
            raw_json = _call_openai(system_prompt, user_prompt)
            logger.debug(f"[llm] Attempt {attempt} raw response: {raw_json[:200]}")

            data = json.loads(raw_json)
            result = SymptomsAnalysisOutput(**data)
            logger.info(f"[llm] Diagnosis generated successfully on attempt {attempt}.")
            return result

        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning(f"[llm] Attempt {attempt} failed to parse response: {exc}")
            if attempt == 2:
                logger.error("[llm] Both attempts failed — returning fallback response.")
                return _fallback_response(language)

        except Exception as exc:
            logger.error(f"[llm] OpenAI call failed on attempt {attempt}: {exc}")
            if attempt == 2:
                return _fallback_response(language)