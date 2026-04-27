"""
agent/tools/drug/rxnorm_client.py
-----------------------------------
Client for the RxNorm API — US National Library of Medicine.

URL:     https://rxnav.nlm.nih.gov/REST/
Cost:    100% FREE — no API key required
Limit:   No documented rate limit for reasonable use
Purpose: Normalize drug names → RxCUI (RxNorm Concept Unique Identifier)

Why RxCUI?
  RxCUI is the standard identifier for drugs in US clinical systems.
  It's what we pass to the RxNav interaction API to find interactions.
  Example: "warfarin" → RxCUI "11289"

Reference: https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html
"""

from __future__ import annotations
import logging
import time
import requests

from agent.tools.drug.schemas.drug_schemas import DrugInfo, DosageInstruction

logger = logging.getLogger(__name__)

_RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
_REQUEST_TIMEOUT = 8   # seconds
_RETRY_DELAY = 1.0     # seconds between retries


def _get(url: str, params: dict | None = None) -> dict | None:
    """Make a GET request with retry logic. Returns parsed JSON or None."""
    for attempt in range(1, 3):
        try:
            response = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.warning(f"[rxnorm] Timeout on attempt {attempt}: {url}")
        except requests.exceptions.HTTPError as e:
            logger.warning(f"[rxnorm] HTTP error: {e}")
            return None
        except Exception as e:
            logger.warning(f"[rxnorm] Request failed attempt {attempt}: {e}")
        if attempt < 2:
            time.sleep(_RETRY_DELAY)
    return None


def get_rxcui(drug_name: str) -> str | None:
    """
    Look up the RxCUI for a drug name.

    Args:
        drug_name: Drug name (e.g. "warfarin", "ibuprofen")

    Returns:
        RxCUI string (e.g. "11289") or None if not found.
    """
    data = _get(
        f"{_RXNORM_BASE}/rxcui.json",
        params={"name": drug_name, "search": "1"},
    )
    if not data:
        return None

    ids = data.get("idGroup", {}).get("rxnormId", [])
    if ids:
        logger.debug(f"[rxnorm] '{drug_name}' → RxCUI {ids[0]}")
        return ids[0]

    # Try approximate search if exact fails
    data2 = _get(
        f"{_RXNORM_BASE}/approximateTerm.json",
        params={"term": drug_name, "maxEntries": 1},
    )
    if data2:
        candidates = data2.get("approximateGroup", {}).get("candidate", [])
        if candidates:
            rxcui = candidates[0].get("rxcui", "")
            if rxcui:
                logger.debug(f"[rxnorm] Approximate match '{drug_name}' → RxCUI {rxcui}")
                return rxcui

    logger.warning(f"[rxnorm] No RxCUI found for '{drug_name}'")
    return None


def get_drug_name_from_rxcui(rxcui: str) -> str:
    """Get the canonical drug name for a given RxCUI."""
    data = _get(f"{_RXNORM_BASE}/rxcui/{rxcui}/property.json",
                params={"propName": "RxNorm Name"})
    if data:
        props = data.get("propConceptGroup", {}).get("propConcept", [])
        for prop in props:
            if prop.get("propName") == "RxNorm Name":
                return prop.get("propValue", "")
    # Fallback: get name directly
    data2 = _get(f"{_RXNORM_BASE}/rxcui/{rxcui}.json")
    if data2:
        return data2.get("idGroup", {}).get("name", "")
    return ""


def normalize_drug(drug_name: str) -> DrugInfo:
    """
    Normalize a drug name: look it up in RxNorm and return a DrugInfo object.

    Args:
        drug_name: Drug name as extracted from user input.

    Returns:
        DrugInfo with rxcui and normalized_name populated if found.
    """
    rxcui = get_rxcui(drug_name)

    if not rxcui:
        return DrugInfo(
            original_name=drug_name,
            normalized_name=drug_name,
            rxcui="",
            found_in_rxnorm=False,
        )

    canonical_name = get_drug_name_from_rxcui(rxcui) or drug_name

    return DrugInfo(
        original_name=drug_name,
        normalized_name=canonical_name,
        rxcui=rxcui,
        found_in_rxnorm=True,
    )