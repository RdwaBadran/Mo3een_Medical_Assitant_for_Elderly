"""
agent/tools/tools.py
----------------------
Central registry of all medical tools available to the LangChain agent.

Status:
  symptoms_analysis        → REAL (RAG + GPT-4o pipeline)
  drug_interaction_checker → REAL (RxNorm + RxNav + OpenFDA + Groq pipeline)
  lab_report_explanation   → REAL (Groq + local range checker pipeline)
"""

# ── Real implementations ───────────────────────────────────────────────────
from agent.tools.symptoms.symptoms_tool import symptoms_analysis          # noqa: F401
from agent.tools.lab.lab_tool import lab_report_explanation               # noqa: F401
from agent.tools.drug.drug_tool import drug_interaction_checker           # noqa: F401

# ── Exported list — imported by agent/tools/__init__.py → agent.py ─────────
ALL_TOOLS = [
    symptoms_analysis,
    drug_interaction_checker,
    lab_report_explanation,
]