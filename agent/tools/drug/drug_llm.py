"""
agent/tools/drug/drug_llm.py
------------------------------
Uses Groq (free tier) to synthesize a patient-friendly drug interaction
report from the raw API data collected by rxnorm_client, rxnav_client,
and openfda_client.

Input:  DrugInteractionReport (partially filled — drugs + raw interactions)
Output: Same object with overall_summary, recommendations, and
        patient-friendly interaction explanations populated.

Language: injected via the prompt — responds in Arabic or English
          based on the detected language from the user's input.
"""

from __future__ import annotations
import os
import json
import logging
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from agent.tools.drug.schemas.drug_schemas import DrugInteractionReport, DrugInteraction
from agent.tools.drug.prompts.drug_prompts import build_synthesis_prompt

load_dotenv()
logger = logging.getLogger(__name__)

_groq_llm: ChatGroq | None = None


def _get_groq():
    global _groq_llm
    if _groq_llm is None:
        from agent.llm_factory import get_groq_llm
        # Drug interaction requires strict reasoning, using 70B
        _groq_llm = get_groq_llm(model_name="llama-3.3-70b-versatile", temperature=0.2)
    return _groq_llm


def _build_drugs_list(report: DrugInteractionReport) -> str:
    """Serialize drug list for the prompt."""
    lines = []
    for drug in report.drugs:
        name = drug.normalized_name or drug.original_name
        status = "✓ Found in RxNorm" if drug.found_in_rxnorm else "⚠ Not found in RxNorm"
        lines.append(f"- {name} ({status})")
    return "\n".join(lines)


def _build_interactions_json(interactions: list[DrugInteraction]) -> str:
    """Serialize interactions for the prompt."""
    if not interactions:
        return "No interactions found in the RxNav database."
    data = [
        {
            "drug_1": ix.drug_1,
            "drug_2": ix.drug_2,
            "severity": ix.severity,
            "description": ix.description,
            "source": ix.source,
        }
        for ix in interactions
    ]
    return json.dumps(data, ensure_ascii=False, indent=2)


def _build_severity_summary(interactions: list[DrugInteraction]) -> str:
    """Build a one-line severity overview for the prompt."""
    if not interactions:
        return "No interactions detected."
    from collections import Counter
    counts = Counter(ix.severity for ix in interactions)
    parts = []
    for severity in ["contraindicated", "serious", "moderate", "minor", "unknown"]:
        if counts[severity]:
            parts.append(f"{counts[severity]} {severity}")
    return ", ".join(parts) if parts else "interactions of unspecified severity found"


def _fallback_summary(report: DrugInteractionReport, language: str) -> DrugInteractionReport:
    """Safe fallback when Groq call fails."""
    if language == "ar":
        report.overall_summary = (
            "حدث خطأ أثناء إنشاء التقرير. يُرجى استشارة طبيبك أو الصيدلاني."
        )
        report.recommendations = ["يُرجى مراجعة طبيبك أو الصيدلاني بشأن هذه الأدوية."]
    else:
        report.overall_summary = (
            "An error occurred while generating the report. "
            "Please consult your doctor or pharmacist."
        )
        report.recommendations = [
            "Please consult your doctor or pharmacist about these medications."
        ]
    return report


def synthesize_report(
    report: DrugInteractionReport,
    language: str = "en",
) -> DrugInteractionReport:
    """
    Use Groq to generate:
    - overall_summary: plain-language overview
    - recommendations: concrete action items
    - patient-friendly explanations for each interaction

    Args:
        report:   DrugInteractionReport with drugs and raw interactions populated.
        language: ISO 639-1 language code ('ar' or 'en').

    Returns:
        The same report object with summary, recommendations, and
        patient-friendly interaction descriptions added.
    """
    drugs_list = _build_drugs_list(report)
    interactions_json = _build_interactions_json(report.interactions)
    severity_summary = _build_severity_summary(report.interactions)

    system_prompt, user_prompt = build_synthesis_prompt(
        drugs_list=drugs_list,
        interactions_json=interactions_json,
        severity_summary=severity_summary,
        language=language,
    )

    for attempt in range(1, 3):
        try:
            llm = _get_groq()
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])

            raw = response.content.strip()
            start = raw.find('{')
            end = raw.rfind('}')
            if start == -1 or end == -1:
                raise ValueError("No JSON object in response")
            data = json.loads(raw[start:end + 1])

            report.overall_summary = data.get("overall_summary", "")
            report.recommendations = data.get("recommendations", [])
            report.disclaimer = data.get(
                "disclaimer",
                (
                    "This information is for educational purposes only. "
                    "Always consult your pharmacist or doctor before changing "
                    "or combining medications."
                ),
            )

            # Merge patient-friendly explanations back into interaction objects
            explanation_map = {
                (e.get("drug_1", "").lower(), e.get("drug_2", "").lower()): e.get("patient_explanation", "")
                for e in data.get("interaction_explanations", [])
            }

            for ix in report.interactions:
                key1 = (ix.drug_1.lower(), ix.drug_2.lower())
                key2 = (ix.drug_2.lower(), ix.drug_1.lower())
                patient_text = explanation_map.get(key1) or explanation_map.get(key2, "")
                if patient_text:
                    ix.description = patient_text   # replace raw API text with patient-friendly text

            report.language = language
            logger.info(f"[drug_llm] Report synthesized on attempt {attempt}.")
            return report

        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning(f"[drug_llm] Attempt {attempt} parse failed: {exc}")
            if attempt == 2:
                return _fallback_summary(report, language)
        except Exception as exc:
            logger.error(f"[drug_llm] Attempt {attempt} Groq call failed: {exc}")
            if attempt == 2:
                return _fallback_summary(report, language)