# agent/tools/drug/prompts/__init__.py
from agent.tools.drug.prompts.drug_prompts import (
    build_extraction_prompt,
    build_synthesis_prompt,
)

__all__ = ["build_extraction_prompt", "build_synthesis_prompt"]