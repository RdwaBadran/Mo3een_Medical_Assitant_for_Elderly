"""
evaluation/__init__.py
------------------------
Mo3een Evaluation Module — CSEDB Clinical Safety-Effectiveness Framework.

A dual-track evaluation system combining LangSmith's technical evaluators
with the CSEDB paper's clinical safety and effectiveness methodology.

Entry point:
    python -m evaluation.run

Components:
    - metric_registry: Source of truth for all 36 metrics
    - targets/: LangSmith-compatible wrappers for Mo3een tools
    - evaluators/: Safety, effectiveness, LLM judge, and technical evaluators
    - dataset/: Test case generation and LangSmith upload
    - aggregator: Weighted scoring formula (CSEDB Eq. 2)
    - run: Pipeline orchestrator
"""

__version__ = "1.0.0"
