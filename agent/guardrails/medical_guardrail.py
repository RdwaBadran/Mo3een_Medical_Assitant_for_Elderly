"""
agent/guardrails/medical_guardrail.py
---------------------------------------
Lightweight guardrail layer that runs BEFORE the symptoms pipeline.

Two responsibilities:
  1. Medical relevance check — is this query actually about health/symptoms?
  2. Language detection — what language did the user write in?

Both use fast, cheap operations so the guardrail adds minimal latency:
  - Language detection: langdetect library (local, instant, free)
  - Medical classification: Groq (fast inference, uses existing GROQ_API_KEY)

The guardrail is called by symptoms_tool.py before any RAG or OpenAI calls.
If the query is not medical, it short-circuits and returns a polite refusal
without spending any OpenAI tokens.
"""

from __future__ import annotations
import os
import logging
from dotenv import load_dotenv

from langdetect import detect, LangDetectException
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from agent.tools.symptoms.prompts.diagnosis_prompt import (
    GUARDRAIL_SYSTEM_PROMPT,
    GUARDRAIL_USER_TEMPLATE,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ── Refusal messages by language ──────────────────────────────────────────────
_REFUSALS = {
    "ar": (
        "أنا مساعد طبي ولا أستطيع المساعدة إلا في الاستفسارات المتعلقة بالصحة والأعراض المرضية. "
        "هل تعاني من أعراض صحية تود مناقشتها؟"
    ),
    "en": (
        "I'm a medical assistant and can only help with health-related questions about symptoms "
        "or medical concerns. Is there a health issue I can help you with?"
    ),
    "default": (
        "I'm a medical assistant and can only help with health-related questions."
    ),
}

# ── Groq client for guardrail classification (cheap + fast) ──────────────────
_groq_llm: ChatGroq | None = None


def _get_groq_llm():
    global _groq_llm
    if _groq_llm is None:
        from agent.llm_factory import get_groq_llm
        _groq_llm = get_groq_llm(model_name="llama-3.1-8b-instant", temperature=0)
    return _groq_llm


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    """
    Detect the language of the input text.

    Args:
        text: The user's raw input string.

    Returns:
        ISO 639-1 language code (e.g. 'ar', 'en', 'fr').
        Returns 'en' as fallback if detection fails.
    """
    try:
        lang = detect(text)
        logger.debug(f"[guardrail] Detected language: {lang}")
        return lang
    except LangDetectException:
        logger.warning("[guardrail] Language detection failed — defaulting to 'en'")
        return "en"


def is_medical_query(query: str) -> bool:
    """
    Use the Groq LLM to classify whether the query is medical.
    Uses the cheapest/fastest model (llama3-8b) — not GPT-4o.
    Returns True if medical, False if not.

    Falls back to True (allow) if the classification fails, to avoid
    blocking legitimate queries due to transient API errors.

    Args:
        query: The user's raw input string.

    Returns:
        bool: True if the query is medical, False otherwise.
    """
    try:
        llm = _get_groq_llm()
        prompt = GUARDRAIL_USER_TEMPLATE.format(query=query)
        response = llm.invoke([
            SystemMessage(content=GUARDRAIL_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        classification = response.content.strip().upper()
        logger.debug(f"[guardrail] Classification: {classification}")
        return classification == "MEDICAL"
    except Exception as exc:
        logger.warning(f"[guardrail] Classification failed: {exc} — defaulting to ALLOW")
        return True   # fail open: don't block the user on API errors


def get_refusal_message(language: str) -> str:
    """
    Return a polite refusal message in the user's language.

    Args:
        language: ISO 639-1 language code.

    Returns:
        Refusal message string.
    """
    return _REFUSALS.get(language, _REFUSALS["default"])


class GuardrailResult:
    """
    Result object returned by run_guardrails().
    Bundles the pass/fail decision with detected language.
    """

    def __init__(self, passed: bool, language: str, refusal_message: str = ""):
        self.passed = passed                   # True = proceed, False = short-circuit
        self.language = language               # detected ISO language code
        self.refusal_message = refusal_message # only populated when passed=False


def run_guardrails(query: str) -> GuardrailResult:
    """
    Run all guardrail checks on a user query.

    Steps:
      1. Detect language
      2. Check if the query is medically relevant

    Args:
        query: The raw user query string.

    Returns:
        GuardrailResult with .passed, .language, and optionally .refusal_message.
    """
    # Step 1: language detection (local, instant)
    language = detect_language(query)

    # Step 2: medical relevance check (Groq, fast)
    if not is_medical_query(query):
        return GuardrailResult(
            passed=False,
            language=language,
            refusal_message=get_refusal_message(language),
        )

    return GuardrailResult(passed=True, language=language)