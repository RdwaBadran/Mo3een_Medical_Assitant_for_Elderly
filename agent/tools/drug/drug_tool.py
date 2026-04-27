"""
agent/tools/drug/drug_tool.py
-------------------------------
The real drug_interaction_checker LangChain @tool.

Full pipeline (5 steps — 100% free):
  1. extract_drug_names()  — Groq extracts individual drug names from any text
  2. normalize_drug()      — RxNorm API: drug name → RxCUI (free, no key)
  3. get_dosage_info()     — OpenFDA API: dosage form, instructions, warnings (free)
  4. get_interactions()    — RxNav API: drug-drug interactions with severity (free)
  5. synthesize_report()   — Groq writes patient-friendly report in Arabic or English

All three APIs (RxNorm, RxNav, OpenFDA) are official US government
medical databases — completely free, no API key required, used in
real clinical systems.

Language: auto-detected from the user's input.
"""

from __future__ import annotations
import logging
from langchain_core.tools import tool

from agent.guardrails.medical_guardrail import detect_language
from agent.tools.drug.drug_name_extractor import extract_drug_names
from agent.tools.drug.rxnorm_client import normalize_drug
from agent.tools.drug.rxnav_client import get_interactions
from agent.tools.drug.openfda_client import get_dosage_info
from agent.tools.drug.drug_llm import synthesize_report
from agent.tools.drug.schemas.drug_schemas import DrugInteractionReport

logger = logging.getLogger(__name__)


@tool
def drug_interaction_checker(drugs: str, language: str = None) -> str:
    """
    Check for interactions between two or more drugs and provide medication instructions.

    Use this tool whenever the user:
    - Asks if two or more medications can be taken together
    - Asks about drug combinations or polypharmacy
    - Wants to know the side effects of combining specific drugs
    - Asks for instructions on how to take a specific medication
    - Mentions medication names and asks if they are safe together

    The tool will:
    - Identify all drug names mentioned in the input
    - Check interactions using official US medical databases (RxNorm + RxNav)
    - Rate interaction severity (contraindicated / serious / moderate / minor)
    - Provide dosage information from the FDA drug label database
    - Explain everything in simple, patient-friendly language

    Supports Arabic ('ar') and English ('en') responses automatically.

    Args:
        drugs: Free text containing one or more drug names. Can be in any format,
               e.g. "Can I take warfarin and ibuprofen?" or "aspirin, metformin, lisinopril"
        language: Optional language override ('ar' or 'en'). Auto-detected if not provided.

    Returns:
        A fully formatted drug interaction report in the patient's language.
    """
    logger.info(f"[drug_tool] Received: {drugs[:100]}...")

    # ── Step 0: Language detection ───────────────────────────────────────────
    if not language:
        language = detect_language(drugs)
    logger.info(f"[drug_tool] Language: {language}")

    # ── Step 1: Extract drug names (Groq) ────────────────────────────────────
    extracted = extract_drug_names(drugs)

    if not extracted.drugs:
        if language == "ar":
            return (
                "لم أتمكن من تحديد أسماء الأدوية في النص المقدم. "
                "يُرجى ذكر أسماء الأدوية بوضوح، مثال: 'هل يمكنني تناول الأسبرين والوارفارين معاً؟'"
            )
        return (
            "I could not identify any drug names in your message. "
            "Please mention the drug names clearly, e.g. "
            "'Can I take aspirin and warfarin together?'"
        )

    if len(extracted.drugs) == 1:
        # Single drug — provide information only, no interaction check needed
        drug_name = extracted.drugs[0]
        logger.info(f"[drug_tool] Single drug detected: {drug_name} — providing info only.")

        drug_info = normalize_drug(drug_name)
        drug_info.dosage = get_dosage_info(drug_info.normalized_name or drug_name)

        report = DrugInteractionReport(
            drugs=[drug_info],
            interactions=[],
            has_serious_interactions=False,
            language=language,
        )
        report = synthesize_report(report, language=language)
        return report.to_markdown()

    logger.info(f"[drug_tool] Drugs extracted: {extracted.drugs}")

    # ── Step 2 & 3: Normalize via RxNorm + get dosage from OpenFDA ───────────
    drug_infos = []
    for drug_name in extracted.drugs:
        logger.debug(f"[drug_tool] Normalizing: {drug_name}")
        info = normalize_drug(drug_name)
        info.dosage = get_dosage_info(info.normalized_name or drug_name)
        drug_infos.append(info)

    # ── Step 4: Get interactions from RxNav ───────────────────────────────────
    rxcuis = [d.rxcui for d in drug_infos if d.rxcui]
    drug_name_map = {d.rxcui: (d.normalized_name or d.original_name) for d in drug_infos if d.rxcui}

    interactions = []
    if len(rxcuis) >= 2:
        interactions = get_interactions(rxcuis, drug_name_map)
    else:
        logger.warning(
            f"[drug_tool] Only {len(rxcuis)} drugs resolved in RxNorm — "
            "cannot check interactions for unrecognized drugs."
        )

    # Determine if any serious interactions exist
    serious_severities = {"contraindicated", "serious"}
    has_serious = any(ix.severity in serious_severities for ix in interactions)

    # ── Step 5: Synthesize with Groq (patient-friendly, language-aware) ───────
    report = DrugInteractionReport(
        drugs=drug_infos,
        interactions=interactions,
        has_serious_interactions=has_serious,
        language=language,
    )
    report = synthesize_report(report, language=language)

    result = report.to_markdown()
    logger.info("[drug_tool] Report complete and formatted.")
    return result