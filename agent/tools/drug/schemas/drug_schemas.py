"""
agent/tools/drug/schemas/drug_schemas.py
------------------------------------------
Pydantic v2 schemas for the drug interaction checker pipeline.

Data flows:
  Raw drug name string
    → DrugNameList (extracted by Groq)
    → DrugInfo (enriched with RxCUI + OpenFDA dosage)
    → DrugInteraction (from RxNav API)
    → DrugInteractionReport (final structured output)
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Extracted drug names from raw text
# ══════════════════════════════════════════════════════════════════════════════

class DrugNameList(BaseModel):
    """Output from drug_name_extractor.py — the Groq extraction step."""
    drugs: list[str] = Field(
        default_factory=list,
        description="List of individual drug names extracted from the user's input.",
    )
    extraction_note: str = Field(
        default="",
        description="Any note from the extractor (e.g. 'brand name detected').",
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Normalized drug with RxCUI + dosage info
# ══════════════════════════════════════════════════════════════════════════════

class DosageInstruction(BaseModel):
    """Dosage/administration information from OpenFDA."""
    route: str = Field(default="", description="Route of administration (oral, IV, etc.)")
    dosage_form: str = Field(default="", description="Form (tablet, capsule, syrup, etc.)")
    dosage_text: str = Field(default="", description="Free-text dosage instructions from FDA label.")
    warnings: list[str] = Field(default_factory=list, description="Key warnings for this drug.")
    contraindications: str = Field(default="", description="Known contraindications.")


class DrugInfo(BaseModel):
    """One drug, normalized and enriched."""
    original_name: str = Field(..., description="Name as typed by the user.")
    normalized_name: str = Field(default="", description="Canonical name from RxNorm.")
    rxcui: str = Field(default="", description="RxNorm Concept Unique Identifier.")
    found_in_rxnorm: bool = Field(default=False)
    dosage: DosageInstruction = Field(default_factory=DosageInstruction)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — One drug-drug interaction pair from RxNav
# ══════════════════════════════════════════════════════════════════════════════

class DrugInteraction(BaseModel):
    """One interaction between two drugs."""
    drug_1: str
    drug_2: str
    severity: Literal["contraindicated", "serious", "moderate", "minor", "unknown"] = "unknown"
    description: str = Field(default="", description="Clinical description of the interaction.")
    source: str = Field(default="RxNav/NLM", description="Data source.")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL OUTPUT — the complete drug interaction report
# ══════════════════════════════════════════════════════════════════════════════

class DrugInteractionReport(BaseModel):
    """
    The complete structured output from the drug interaction pipeline.
    This is what drug_tool.py returns after the full pipeline.
    """
    drugs: list[DrugInfo]
    interactions: list[DrugInteraction]
    has_serious_interactions: bool = False
    overall_summary: str = Field(default="", description="Plain-language overall summary.")
    recommendations: list[str] = Field(default_factory=list)
    language: str = Field(default="en")
    disclaimer: str = Field(
        default=(
            "This information is for educational purposes only. Always consult "
            "your pharmacist or doctor before changing or combining medications."
        )
    )

    def to_markdown(self) -> str:
        """Format the full report as markdown."""
        if self.language == "ar":
            return self._format_arabic()
        return self._format_english()

    def _severity_emoji(self, severity: str) -> str:
        return {
            "contraindicated": "🚫",
            "serious":         "🔴",
            "moderate":        "🟠",
            "minor":           "🟡",
            "unknown":         "⚪",
        }.get(severity, "⚪")

    def _format_english(self) -> str:
        lines = ["## 💊 Drug Interaction Report\n"]

        # Alert banner for serious interactions
        if self.has_serious_interactions:
            lines.append(
                "> 🚨 **IMPORTANT:** One or more serious interactions were found. "
                "Please consult your doctor or pharmacist before taking these medications together.\n"
            )

        # Overall summary
        lines.append("### 📋 Summary")
        lines.append(self.overall_summary)
        lines.append("")

        # Interactions section
        if self.interactions:
            lines.append("### ⚠️ Detected Interactions")
            for ix in self.interactions:
                emoji = self._severity_emoji(ix.severity)
                lines.append(
                    f"**{emoji} {ix.drug_1} ↔ {ix.drug_2}** — "
                    f"Severity: **{ix.severity.upper()}**"
                )
                if ix.description:
                    lines.append(f"> {ix.description}")
                lines.append(f"> *Source: {ix.source}*")
                lines.append("")
        else:
            lines.append("### ✅ No Known Interactions Found")
            lines.append(
                "No interactions were found between these medications in the RxNav database. "
                "This does not guarantee they are safe to combine — always confirm with your pharmacist."
            )
            lines.append("")

        # Per-drug information
        lines.append("### 📖 Medication Information")
        for drug in self.drugs:
            name = drug.normalized_name or drug.original_name
            lines.append(f"#### 💊 {name}")
            d = drug.dosage
            if d.dosage_form:
                lines.append(f"- **Form:** {d.dosage_form}")
            if d.route:
                lines.append(f"- **Route:** {d.route}")
            if d.dosage_text:
                lines.append(f"- **Dosage guidance:** {d.dosage_text}")
            if d.warnings:
                lines.append("- **Key warnings:**")
                for w in d.warnings[:3]:   # limit to 3 most important
                    lines.append(f"  - {w}")
            if d.contraindications:
                lines.append(f"- **Contraindications:** {d.contraindications}")
            if not drug.found_in_rxnorm:
                lines.append(f"  *(Drug not found in RxNorm database — name may need verification)*")
            lines.append("")

        # Recommendations
        if self.recommendations:
            lines.append("### ✅ Recommendations")
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        lines.append(f"---\n*{self.disclaimer}*")
        return "\n".join(lines)

    def _format_arabic(self) -> str:
        lines = ["## 💊 تقرير التفاعلات الدوائية\n"]

        if self.has_serious_interactions:
            lines.append(
                "> 🚨 **تنبيه هام:** تم اكتشاف تفاعل دوائي خطير. "
                "يُرجى استشارة طبيبك أو الصيدلاني قبل تناول هذه الأدوية معاً.\n"
            )

        lines.append("### 📋 الملخص العام")
        lines.append(self.overall_summary)
        lines.append("")

        severity_ar = {
            "contraindicated": "مُضاد — لا يجوز الجمع",
            "serious":         "خطير",
            "moderate":        "متوسط",
            "minor":           "بسيط",
            "unknown":         "غير محدد",
        }

        if self.interactions:
            lines.append("### ⚠️ التفاعلات المكتشفة")
            for ix in self.interactions:
                emoji = self._severity_emoji(ix.severity)
                lines.append(
                    f"**{emoji} {ix.drug_1} ↔ {ix.drug_2}** — "
                    f"الخطورة: **{severity_ar.get(ix.severity, ix.severity)}**"
                )
                if ix.description:
                    lines.append(f"> {ix.description}")
                lines.append(f"> *المصدر: {ix.source}*")
                lines.append("")
        else:
            lines.append("### ✅ لم يتم اكتشاف تفاعلات معروفة")
            lines.append(
                "لم يتم العثور على تفاعلات دوائية معروفة بين هذه الأدوية. "
                "هذا لا يعني بالضرورة أنها آمنة للجمع — يُرجى التأكد من الصيدلاني."
            )
            lines.append("")

        lines.append("### 📖 معلومات الأدوية")
        for drug in self.drugs:
            name = drug.normalized_name or drug.original_name
            lines.append(f"#### 💊 {name}")
            d = drug.dosage
            if d.dosage_form:
                lines.append(f"- **شكل الدواء:** {d.dosage_form}")
            if d.route:
                lines.append(f"- **طريقة التناول:** {d.route}")
            if d.dosage_text:
                lines.append(f"- **إرشادات الجرعة:** {d.dosage_text}")
            if d.warnings:
                lines.append("- **تحذيرات أساسية:**")
                for w in d.warnings[:3]:
                    lines.append(f"  - {w}")
            if d.contraindications:
                lines.append(f"- **موانع الاستخدام:** {d.contraindications}")
            lines.append("")

        if self.recommendations:
            lines.append("### ✅ التوصيات")
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        lines.append(
            f"---\n*تنبيه: هذه المعلومات لأغراض تعليمية فقط. "
            f"استشر طبيبك أو الصيدلاني دائماً قبل تغيير أو دمج الأدوية.*"
        )
        return "\n".join(lines)