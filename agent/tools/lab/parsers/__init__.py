# agent/tools/lab/parsers/__init__.py
from agent.tools.lab.parsers.pdf_parser import parse_pdf
from agent.tools.lab.parsers.ocr_parser import parse_image

__all__ = ["parse_pdf", "parse_image"]