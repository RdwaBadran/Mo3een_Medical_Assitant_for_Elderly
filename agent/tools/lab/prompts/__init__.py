# agent/tools/lab/prompts/__init__.py
from agent.tools.lab.prompts.lab_prompts import (
    build_extraction_prompt,
    build_explanation_prompt,
)

__all__ = ["build_extraction_prompt", "build_explanation_prompt"]