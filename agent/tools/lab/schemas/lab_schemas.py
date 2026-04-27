"""
agent/tools/lab/schemas/lab_schemas.py
----------------------------------------
Pydantic v2 schemas for the lab report explanation pipeline.

Input:  LabReportInput  — raw text from user or parsed from file
Output: LabParameter    — one checked parameter with explanation
        LabReport       — the full structured report
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, field_validator


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION — what Groq returns when parsing raw text
# ══════════════════════════════════════════════════════════════════════════════

class RawParameter(BaseModel):
    """
    One lab parameter as extracted from raw text by the Groq extractor.
    Name and unit are as-found — normalisation happens in range_checker.py.
    """
    name: str = Field(..., description="Parameter name as found in the report (e.g. 'HbA1c', 'WBC').")
    value: float = Field(..., description="Numeric value.")
    unit: str = Field(default="", description="Unit as found (e.g. '%', 'mg/dL', 'x10³/µL').")


class ExtractionResult(BaseModel):
    """Output from parameter_extractor.py."""
    parameters: list[RawParameter] = Field(
        default_factory=list,
        description="All lab parameters successfully extracted from the raw text.",
    )
    extraction_note: str = Field(
        default="",
        description="Any note from the extractor (e.g. 'some values could not be parsed').",
    )


# ══════════════════════════════════════════════════════════════════════════════
# CHECKED PARAMETER — after range_checker.py processes it
# ══════════════════════════════════════════════════════════════════════════════

class LabParameter(BaseModel):
    """
    A fully enriched lab parameter: extracted value + range check + LLM explanation.
    """
    name: str
    full_name: str = ""
    value: float
    unit: str
    normal_min: float
    normal_max: float
    status: Literal["normal", "borderline_high", "high", "low", "critical_high", "critical_low", "unknown"]
    deviation_percent: float = Field(
        default=0.0,
        description="How far from the nearest normal boundary, as a percentage.",
    )
    panel: str = Field(default="", description="Which test panel this belongs to (e.g. CBC, Lipid Panel).")
    explanation: str = Field(default="", description="Plain-language explanation — filled by lab_llm.py.")
    risk_note: str = Field(default="", description="Risk or clinical significance — filled by lab_llm.py.")
    clinical_note: str = Field(default="", description="Reference note from normal_ranges.json.")

    @property
    def status_emoji(self) -> str:
        return {
            "normal":        "🟢",
            "borderline_high": "🟡",
            "high":          "🔴",
            "low":           "🔵",
            "critical_high": "🚨",
            "critical_low":  "🚨",
            "unknown":       "⚪",
        }.get(self.status, "⚪")

    @property
    def status_label_en(self) -> str:
        return {
            "normal":        "Normal",
            "borderline_high": "Borderline High",
            "high":          "High",
            "low":           "Low",
            "critical_high": "CRITICAL HIGH",
            "critical_low":  "CRITICAL LOW",
            "unknown":       "Not in reference range",
        }.get(self.status, "Unknown")

    @property
    def status_label_ar(self) -> str:
        return {
            "normal":        "طبيعي",
            "borderline_high": "مرتفع حدياً",
            "high":          "مرتفع",
            "low":           "منخفض",
            "critical_high": "مرتفع بشكل حرج",
            "critical_low":  "منخفض بشكل حرج",
            "unknown":       "غير موجود في قاعدة البيانات",
        }.get(self.status, "غير معروف")


# ══════════════════════════════════════════════════════════════════════════════
# FULL REPORT — the final output
# ══════════════════════════════════════════════════════════════════════════════

class LabReport(BaseModel):
    """
    The complete structured lab report explanation.
    This is what lab_tool.py returns after the full pipeline.
    """
    parameters: list[LabParameter]
    overall_assessment: str = Field(
        ..., description="One-paragraph summary of the patient's overall lab picture."
    )
    urgent_flags: list[str] = Field(
        default_factory=list,
        description="Parameters in critical range that require immediate medical attention.",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Concrete suggested next steps for the patient.",
    )
    language: str = Field(default="en")
    disclaimer: str = Field(
        default=(
            "This analysis is AI-generated and is not a substitute for professional "
            "medical interpretation. Please consult your doctor to discuss these results."
        )
    )

    def to_markdown(self) -> str:
        """Convert the full report to a formatted markdown string."""
        if self.language == "ar":
            return self._format_arabic()
        return self._format_english()

    def _format_english(self) -> str:
        lines = ["## 🧪 Lab Report Analysis\n"]

        # Urgent flags first
        if self.urgent_flags:
            lines.append("### 🚨 URGENT — Requires Immediate Medical Attention")
            for flag in self.urgent_flags:
                lines.append(f"- {flag}")
            lines.append("")

        # Overall assessment
        lines.append("### 📋 Overall Assessment")
        lines.append(self.overall_assessment)
        lines.append("")

        # Group parameters by panel
        panels: dict[str, list[LabParameter]] = {}
        for p in self.parameters:
            panel = p.panel or "Other"
            panels.setdefault(panel, []).append(p)

        for panel_name, params in panels.items():
            lines.append(f"### 📊 {panel_name}")
            for p in params:
                lines.append(
                    f"**{p.status_emoji} {p.full_name or p.name}** — "
                    f"`{p.value} {p.unit}` | Normal: `{p.normal_min} – {p.normal_max} {p.unit}` | "
                    f"**{p.status_label_en}**"
                )
                if p.explanation:
                    lines.append(f"> {p.explanation}")
                if p.risk_note and p.status not in ("normal",):
                    lines.append(f"> ⚠️ {p.risk_note}")
                lines.append("")

        # Recommendations
        if self.recommendations:
            lines.append("### ✅ Recommended Next Steps")
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        lines.append(f"---\n*{self.disclaimer}*")
        return "\n".join(lines)

    def _format_arabic(self) -> str:
        lines = ["## 🧪 تحليل نتائج التحاليل الطبية\n"]

        if self.urgent_flags:
            lines.append("### 🚨 تنبيه عاجل — يستدعي مراجعة الطبيب فوراً")
            for flag in self.urgent_flags:
                lines.append(f"- {flag}")
            lines.append("")

        lines.append("### 📋 التقييم العام")
        lines.append(self.overall_assessment)
        lines.append("")

        panels: dict[str, list[LabParameter]] = {}
        for p in self.parameters:
            panel = p.panel or "أخرى"
            panels.setdefault(panel, []).append(p)

        for panel_name, params in panels.items():
            lines.append(f"### 📊 {panel_name}")
            for p in params:
                lines.append(
                    f"**{p.status_emoji} {p.full_name or p.name}** — "
                    f"`{p.value} {p.unit}` | المعدل الطبيعي: `{p.normal_min} – {p.normal_max} {p.unit}` | "
                    f"**{p.status_label_ar}**"
                )
                if p.explanation:
                    lines.append(f"> {p.explanation}")
                if p.risk_note and p.status not in ("normal",):
                    lines.append(f"> ⚠️ {p.risk_note}")
                lines.append("")

        if self.recommendations:
            lines.append("### ✅ الخطوات الموصى بها")
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        lines.append(f"---\n*{self.disclaimer}*")
        return "\n".join(lines)