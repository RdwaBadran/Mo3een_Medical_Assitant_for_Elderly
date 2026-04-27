"""
agent/tools/drug/rxnav_client.py
----------------------------------
Client for the RxNav Drug Interaction API — US National Library of Medicine.

URL:     https://rxnav.nlm.nih.gov/REST/interaction/
Cost:    100% FREE — no API key required
Purpose: Given a list of RxCUI IDs, return all known drug-drug interactions

This API aggregates interaction data from:
  - ONCHigh (Office of the National Coordinator)
  - DrugBank
  - Clinical Pharmacology
  - Epocrates
  - Lexi-Interact
  - Multum
  - Natural Medicines

Reference: https://lhncbc.nlm.nih.gov/RxNav/APIs/InteractionAPIs.html
"""

from __future__ import annotations
import logging
import time
import requests

from agent.tools.drug.schemas.drug_schemas import DrugInteraction

logger = logging.getLogger(__name__)

_RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST/interaction"
_REQUEST_TIMEOUT = 10
_RETRY_DELAY = 1.0

# Map RxNav severity descriptions to our severity enum
_SEVERITY_MAP = {
    "contraindicated drug combination": "contraindicated",
    "serious - use alternative": "serious",
    "serious": "serious",
    "do not use": "contraindicated",
    "use with caution/monitor": "moderate",
    "moderate": "moderate",
    "use with caution": "moderate",
    "minor": "minor",
    "minimal": "minor",
}


def _get(url: str, params: dict | None = None) -> dict | None:
    """Make a GET request with retry."""
    for attempt in range(1, 3):
        try:
            response = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.warning(f"[rxnav] Timeout attempt {attempt}: {url}")
        except Exception as e:
            logger.warning(f"[rxnav] Request failed attempt {attempt}: {e}")
        if attempt < 2:
            time.sleep(_RETRY_DELAY)
    return None


def _normalise_severity(description: str) -> str:
    """Map a raw severity string to our enum values."""
    desc_lower = description.lower()
    for key, value in _SEVERITY_MAP.items():
        if key in desc_lower:
            return value
    return "unknown"


def get_interactions(rxcuis: list[str], drug_names: dict[str, str]) -> list[DrugInteraction]:
    """
    Fetch all drug-drug interactions for a list of RxCUI IDs.

    Args:
        rxcuis:     List of RxCUI strings (e.g. ["11289", "5640"])
        drug_names: Map from rxcui → canonical drug name for display

    Returns:
        List of DrugInteraction objects. Empty list if no interactions found or API fails.
    """
    if len(rxcuis) < 2:
        logger.info("[rxnav] Need at least 2 drugs to check interactions.")
        return []

    rxcui_str = "+".join(rxcuis)
    data = _get(
        f"{_RXNAV_BASE}/list.json",
        params={"rxcuis": rxcui_str},
    )

    if not data:
        logger.warning("[rxnav] API returned no data.")
        return []

    interactions: list[DrugInteraction] = []
    groups = data.get("fullInteractionTypeGroup", [])

    for group in groups:
        source = group.get("sourceName", "RxNav/NLM")
        interaction_types = group.get("fullInteractionType", [])

        for itype in interaction_types:
            pairs = itype.get("interactionPair", [])
            for pair in pairs:
                concepts = pair.get("interactionConcept", [])
                if len(concepts) < 2:
                    continue

                rxcui_1 = concepts[0].get("minConceptItem", {}).get("rxcui", "")
                rxcui_2 = concepts[1].get("minConceptItem", {}).get("rxcui", "")

                drug_1 = drug_names.get(rxcui_1, rxcui_1)
                drug_2 = drug_names.get(rxcui_2, rxcui_2)

                raw_severity = pair.get("severity", "")
                severity = _normalise_severity(raw_severity)
                description = pair.get("description", "")

                # Avoid duplicate interactions (A↔B same as B↔A)
                key = tuple(sorted([drug_1, drug_2]))
                already_added = any(
                    tuple(sorted([ix.drug_1, ix.drug_2])) == key
                    for ix in interactions
                )
                if not already_added:
                    interactions.append(DrugInteraction(
                        drug_1=drug_1,
                        drug_2=drug_2,
                        severity=severity,
                        description=description,
                        source=source,
                    ))

    logger.info(f"[rxnav] Found {len(interactions)} interactions.")
    return interactions