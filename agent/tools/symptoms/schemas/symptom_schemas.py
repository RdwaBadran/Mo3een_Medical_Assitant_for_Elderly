"""
agent/tools/symptoms/schemas/symptom_schemas.py
-------------------------------------------------
Pydantic v2 schemas that define the exact shape of data
flowing INTO and OUT OF the symptoms analysis pipeline.

Why this matters:
- Enforces that the symptoms string is never empty or too short
- Gives the LLM a strict output structure to fill
- Makes the response predictable for the frontend (no surprises)
- Enables future API versioning without breaking changes
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, field_validator


# ══════════════════════════════════════════════════════════════════════════════
# INPUT SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

class SymptomsInput(BaseModel):
    """
    Validated input to the symptoms_analysis tool.
    Raises ValidationError (caught by the tool) if constraints are violated.
    """

    symptoms: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Plain-text description of the patient's symptoms.",
    )

    detected_language: str = Field(
        default="en",
        description="ISO 639-1 language code detected from the user's input (e.g. 'ar', 'en').",
    )

    @field_validator("symptoms")
    @classmethod
    def symptoms_must_be_medical(cls, v: str) -> str:
        """Strip and normalise whitespace."""
        return " ".join(v.split())


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

class PossibleCondition(BaseModel):
    """One differential diagnosis entry."""

    name: str = Field(..., description="Name of the condition (e.g. 'Tension Headache').")
    rationale: str = Field(..., description="Why this condition fits the reported symptoms.")
    urgency: Literal["low", "medium", "high", "emergency"] = Field(
        ...,
        description="How urgently the patient should seek care for this condition.",
    )


class SymptomsAnalysisOutput(BaseModel):
    """
    Structured output from the symptoms analysis pipeline.
    The LLM is instructed to produce JSON matching this schema exactly.
    The tool validates the JSON before returning it to the agent.
    """

    possible_conditions: list[PossibleCondition] = Field(
        ...,
        min_length=1,
        max_length=3,
        description="Between 1 and 3 most likely differential diagnoses.",
    )

    red_flags: list[str] = Field(
        default_factory=list,
        description="Symptoms that — if they appear or worsen — require immediate emergency care.",
    )

    recommended_next_steps: list[str] = Field(
        ...,
        min_length=1,
        description="Concrete actions the patient should take (e.g. 'Visit a GP within 24 hours').",
    )

    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="Confidence in the analysis based on available RAG context.",
    )

    disclaimer: str = Field(
        default=(
            "This is an AI-generated assessment and is not a substitute for "
            "professional medical advice. Please consult a qualified healthcare "
            "provider for an accurate diagnosis and treatment plan."
        ),
        description="Standard medical disclaimer always included in the response.",
    )

    def to_readable_text(self, language: str = "en") -> str:
        """
        Convert the structured output to a human-readable string.
        Supports English ('en') and Arabic ('ar').
        Called by symptoms_tool.py before returning to the agent.
        """
        if language == "ar":
            return self._format_arabic()
        return self._format_english()

    def _format_english(self) -> str:
        lines = ["**🔬 Symptoms Analysis**\n"]

        lines.append("**Possible Conditions:**")
        for i, cond in enumerate(self.possible_conditions, 1):
            urgency_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "emergency": "🔴"}
            lines.append(
                f"  {i}. **{cond.name}** {urgency_emoji.get(cond.urgency, '')} "
                f"*(urgency: {cond.urgency})*\n"
                f"     {cond.rationale}"
            )

        if self.red_flags:
            lines.append("\n**⚠️ Red Flags — Seek Emergency Care If:**")
            for flag in self.red_flags:
                lines.append(f"  • {flag}")

        lines.append("\n**✅ Recommended Next Steps:**")
        for step in self.recommended_next_steps:
            lines.append(f"  • {step}")

        lines.append(f"\n**Confidence Level:** {self.confidence.capitalize()}")
        lines.append(f"\n*{self.disclaimer}*")

        return "\n".join(lines)

    def _format_arabic(self) -> str:
        lines = ["**🔬 تحليل الأعراض**\n"]

        lines.append("**الحالات المحتملة:**")
        urgency_ar = {"low": "منخفض", "medium": "متوسط", "high": "مرتفع", "emergency": "طارئ"}
        for i, cond in enumerate(self.possible_conditions, 1):
            urgency_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "emergency": "🔴"}
            lines.append(
                f"  {i}. **{cond.name}** {urgency_emoji.get(cond.urgency, '')} "
                f"*(مستوى الاستعجال: {urgency_ar.get(cond.urgency, cond.urgency)})*\n"
                f"     {cond.rationale}"
            )

        if self.red_flags:
            lines.append("\n**⚠️ علامات تحذيرية — اطلب الرعاية الطارئة فوراً إذا:**")
            for flag in self.red_flags:
                lines.append(f"  • {flag}")

        lines.append("\n**✅ الخطوات الموصى بها:**")
        for step in self.recommended_next_steps:
            lines.append(f"  • {step}")

        lines.append(f"\n**مستوى الثقة:** {self.confidence}")
        lines.append(
            f"\n*تنبيه: هذا تقييم مُوَلَّد بالذكاء الاصطناعي وليس بديلاً عن الاستشارة الطبية المتخصصة. "
            f"يُرجى مراجعة طبيب مؤهل للحصول على تشخيص دقيق وخطة علاج مناسبة.*"
        )

        return "\n".join(lines)