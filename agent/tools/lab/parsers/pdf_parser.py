"""
agent/tools/lab/parsers/pdf_parser.py
---------------------------------------
Extracts text from PDF lab report files.

Strategy (two-layer):
  Layer 1: pypdf  — fast, handles most text-based PDFs
  Layer 2: pdfminer.six — better layout preservation as fallback

Returns a clean text string ready to be passed to parameter_extractor.py.
"""

from __future__ import annotations
import io
import logging

logger = logging.getLogger(__name__)


def _extract_with_pypdf(file_bytes: bytes) -> str:
    """Primary extraction using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        result = "\n\n".join(pages)
        logger.debug(f"[pdf_parser] pypdf extracted {len(result)} chars.")
        return result
    except Exception as exc:
        logger.warning(f"[pdf_parser] pypdf failed: {exc}")
        return ""


def _extract_with_pdfminer(file_bytes: bytes) -> str:
    """Fallback extraction using pdfminer.six for better layout handling."""
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        result = pdfminer_extract(io.BytesIO(file_bytes))
        logger.debug(f"[pdf_parser] pdfminer extracted {len(result)} chars.")
        return result or ""
    except ImportError:
        logger.warning("[pdf_parser] pdfminer.six not installed — skipping fallback.")
        return ""
    except Exception as exc:
        logger.warning(f"[pdf_parser] pdfminer failed: {exc}")
        return ""


def parse_pdf(file_bytes: bytes) -> str:
    """
    Extract text from a PDF file.

    Args:
        file_bytes: Raw bytes of the PDF file.

    Returns:
        Extracted text string, or empty string if extraction fails.
    """
    # Try pypdf first
    text = _extract_with_pypdf(file_bytes)

    # If pypdf yields less than 50 chars, try pdfminer
    if len(text.strip()) < 50:
        logger.info("[pdf_parser] pypdf result too short — trying pdfminer fallback.")
        text = _extract_with_pdfminer(file_bytes)

    if not text.strip():
        logger.warning("[pdf_parser] Both extractors returned empty text.")

    return text.strip()