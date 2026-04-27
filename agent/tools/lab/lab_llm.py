"""
agent/tools/lab/lab_llm.py
----------------------------
Uses Groq (free tier) to generate plain-language explanations
for each lab parameter, plus an overall assessment and recommendations.

Receives the list of checked LabParameter objects from range_checker.py.
Returns the same list with explanation and risk_note fields populated,
plus overall_assessment, recommendations, urgent_flags.
"""

from __future__ import annotations
import os
import json
import logging
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from agent.tools.lab.schemas.lab_schemas import LabParameter, LabReport
from agent.tools.lab.prompts.lab_prompts import build_explanation_prompt

load_dotenv()
logger = logging.getLogger(__name__)

# ── Groq singleton ────────────────────────────────────────────────────────────
_groq_llm: ChatGroq | None = None


def _get_groq():
    global _groq_llm
    if _groq_llm is None:
        from agent.llm_factory import get_groq_llm
        _groq_llm = get_groq_llm(model_name="llama-3.3-70b-versatile", temperature=0.2)
    return _groq_llm


def _build_params_json(params: list[LabParameter]) -> str:
    """
    Serialize checked parameters to a compact JSON string for the LLM prompt.
    Includes only the fields the LLM needs to write explanations.
    """
    data = []
    for p in params:
        data.append({
            "name": p.name,
            "full_name": p.full_name,
            "value": p.value,
            "unit": p.unit,
            "normal_range": f"{p.normal_min}–{p.normal_max} {p.unit}",
            "status": p.status,
            "deviation_percent": p.deviation_percent,
            "clinical_note": p.clinical_note,
        })
    return json.dumps(data, ensure_ascii=False, indent=2)


def _build_fallback_report(
    params: list[LabParameter],
    language: str,
) -> LabReport:
    """Safe fallback when the LLM call fails completely."""
    if language == "ar":
        return LabReport(
            parameters=params,
            overall_assessment="حدث خطأ أثناء تحليل نتائج التحاليل. يُرجى مراجعة طبيبك مباشرة.",
            urgent_flags=[],
            recommendations=["يُرجى مراجعة طبيبك لمناقشة نتائج تحاليلك."],
            language=language,
            disclaimer="هذا تقييم مُوَلَّد بالذكاء الاصطناعي وليس بديلاً عن الاستشارة الطبية.",
        )
    return LabReport(
        parameters=params,
        overall_assessment="An error occurred during analysis. Please consult your doctor directly.",
        urgent_flags=[],
        recommendations=["Please consult your doctor to discuss your lab results."],
        language=language,
        disclaimer=(
            "This is an AI-generated analysis and is not a substitute for "
            "professional medical interpretation."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def generate_explanations(
    checked_params: list[LabParameter],
    language: str = "en",
) -> LabReport:
    """
    Call Groq to generate patient-friendly explanations for each parameter
    and produce an overall assessment.

    Args:
        checked_params: List of LabParameter from range_checker.py
        language:       ISO 639-1 language code ('ar' or 'en')

    Returns:
        LabReport with all parameters populated with explanations,
        plus overall_assessment, recommendations, urgent_flags.
    """
    if not checked_params:
        if language == "ar":
            return LabReport(
                parameters=[],
                overall_assessment="لم يتم العثور على أي قيم مختبرية في النص المقدم.",
                urgent_flags=[],
                recommendations=["يُرجى التأكد من إدخال نتائج التحاليل بشكل صحيح."],
                language=language,
            )
        return LabReport(
            parameters=[],
            overall_assessment="No lab values were found in the provided text.",
            urgent_flags=[],
            recommendations=["Please ensure you have entered your lab results correctly."],
            language=language,
        )

    params_json = _build_params_json(checked_params)
    system_prompt, user_prompt = build_explanation_prompt(params_json, language)

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

            # Merge Groq explanations back into checked_params
            explanation_map: dict[str, dict] = {
                e["name"]: e
                for e in data.get("parameter_explanations", [])
            }

            enriched_params = []
            for p in checked_params:
                exp = explanation_map.get(p.name, {})
                p.explanation = exp.get("explanation", "")
                p.risk_note = exp.get("risk_note", "")
                enriched_params.append(p)

            logger.info(
                f"[lab_llm] Attempt {attempt}: explanations generated for "
                f"{len(enriched_params)} parameters."
            )

            return LabReport(
                parameters=enriched_params,
                overall_assessment=data.get("overall_assessment", ""),
                urgent_flags=data.get("urgent_flags", []),
                recommendations=data.get("recommendations", []),
                language=language,
                disclaimer=data.get(
                    "disclaimer",
                    (
                        "This is an AI-generated analysis and is not a substitute "
                        "for professional medical interpretation."
                    ),
                ),
            )

        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(f"[lab_llm] Attempt {attempt} parse failed: {exc}")
            if attempt == 2:
                return _build_fallback_report(checked_params, language)

        except Exception as exc:
            logger.error(f"[lab_llm] Attempt {attempt} Groq call failed: {exc}")
            if attempt == 2:
                return _build_fallback_report(checked_params, language)