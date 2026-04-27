"""
agent/tools/lab/parsers/ocr_parser.py
----------------------------------------
OCR extraction for scanned/photographed lab reports.
Uses pytesseract + Pillow (both free and local).

Requirements:
  pip install pytesseract pillow
  System: Tesseract OCR binary must be installed
    Windows: https://github.com/UB-Mannheim/tesseract/wiki
    Then add to PATH or set pytesseract.pytesseract.tesseract_cmd

Supports: Arabic + English OCR (tesseract lang packs ara+eng).
"""

from __future__ import annotations
import io
import logging

logger = logging.getLogger(__name__)


def _check_tesseract() -> bool:
    """Return True if tesseract is available."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def parse_image(file_bytes: bytes, language: str = "en") -> str:
    """
    Run OCR on an image file (JPEG, PNG, TIFF, BMP).

    Args:
        file_bytes: Raw bytes of the image file.
        language:   Detected language code ('ar' or 'en') — affects OCR lang pack.

    Returns:
        Extracted text string, or empty string if OCR fails.
    """
    try:
        from PIL import Image
        import pytesseract
    except ImportError as exc:
        logger.error(f"[ocr_parser] Missing dependency: {exc}. Run: pip install pytesseract pillow")
        return ""

    if not _check_tesseract():
        logger.error(
            "[ocr_parser] Tesseract binary not found. "
            "Install from: https://github.com/UB-Mannheim/tesseract/wiki"
        )
        return ""

    try:
        image = Image.open(io.BytesIO(file_bytes))

        # Convert to RGB if needed (handles RGBA PNGs)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        # Choose tesseract language pack
        # ara+eng handles mixed Arabic/English lab reports
        tess_lang = "ara+eng" if language == "ar" else "eng"

        text = pytesseract.image_to_string(image, lang=tess_lang)
        logger.debug(f"[ocr_parser] OCR extracted {len(text)} chars.")
        return text.strip()

    except Exception as exc:
        logger.error(f"[ocr_parser] OCR failed: {exc}")
        return ""