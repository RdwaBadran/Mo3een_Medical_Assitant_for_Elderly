"""
agent/tools/drug/openfda_client.py
------------------------------------
Client for the OpenFDA Drug Label API.

URL:     https://api.fda.gov/drug/label.json
Cost:    100% FREE — no API key needed (1000 requests/day without key)
Purpose: Get dosage form, route, dosage instructions, warnings,
         and contraindications for a drug

Reference: https://open.fda.gov/apis/drug/label/
"""

from __future__ import annotations
import logging
import time
import requests

from agent.tools.drug.schemas.drug_schemas import DosageInstruction

logger = logging.getLogger(__name__)

_FDA_BASE = "https://api.fda.gov/drug/label.json"
_REQUEST_TIMEOUT = 8
_RETRY_DELAY = 1.0

# Maximum characters we extract from any free-text FDA field
_MAX_TEXT_LEN = 400


def _get(params: dict) -> dict | None:
    """Make a GET request to OpenFDA with retry."""
    for attempt in range(1, 3):
        try:
            response = requests.get(_FDA_BASE, params=params, timeout=_REQUEST_TIMEOUT)
            if response.status_code == 404:
                return None   # drug not found — not an error
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.warning(f"[openfda] Timeout attempt {attempt}")
        except requests.exceptions.HTTPError as e:
            logger.warning(f"[openfda] HTTP error: {e}")
            return None
        except Exception as e:
            logger.warning(f"[openfda] Request failed attempt {attempt}: {e}")
        if attempt < 2:
            time.sleep(_RETRY_DELAY)
    return None


def _truncate(text: str, max_len: int = _MAX_TEXT_LEN) -> str:
    """Truncate long FDA label text to a readable length."""
    text = text.strip().replace("\n", " ").replace("  ", " ")
    if len(text) > max_len:
        return text[:max_len].rsplit(" ", 1)[0] + "…"
    return text


def get_dosage_info(drug_name: str) -> DosageInstruction:
    """
    Fetch dosage form, route, instructions, warnings, and contraindications
    from the OpenFDA drug label database.

    Args:
        drug_name: Canonical drug name (e.g. "warfarin", "ibuprofen")

    Returns:
        DosageInstruction object. Fields are empty strings if not found.
    """
    # Search by generic name first
    data = _get({
        "search": f'openfda.generic_name:"{drug_name}"',
        "limit": 1,
    })

    # If not found by generic name, try brand name
    if not data or not data.get("results"):
        data = _get({
            "search": f'openfda.brand_name:"{drug_name}"',
            "limit": 1,
        })

    # Last fallback: free-text search on the substance name
    if not data or not data.get("results"):
        data = _get({
            "search": f'openfda.substance_name:"{drug_name}"',
            "limit": 1,
        })

    if not data or not data.get("results"):
        logger.info(f"[openfda] No label found for '{drug_name}'")
        return DosageInstruction()

    label = data["results"][0]
    openfda = label.get("openfda", {})

    # Extract dosage form and route
    dosage_form = ", ".join(openfda.get("dosage_form", [])) or ""
    route = ", ".join(openfda.get("route", [])) or ""

    # Extract dosage text (from dosage_and_administration section)
    dosage_sections = label.get("dosage_and_administration", [])
    dosage_text = _truncate(dosage_sections[0]) if dosage_sections else ""

    # Extract warnings
    raw_warnings = label.get("warnings", []) + label.get("boxed_warning", [])
    warnings = []
    for w in raw_warnings[:2]:   # max 2 warning blocks
        clean = _truncate(w, max_len=200)
        if clean:
            warnings.append(clean)

    # Extract contraindications
    contra_sections = label.get("contraindications", [])
    contraindications = _truncate(contra_sections[0], max_len=300) if contra_sections else ""

    logger.debug(f"[openfda] Got label for '{drug_name}': form={dosage_form}, route={route}")

    return DosageInstruction(
        route=route,
        dosage_form=dosage_form,
        dosage_text=dosage_text,
        warnings=warnings,
        contraindications=contraindications,
    )