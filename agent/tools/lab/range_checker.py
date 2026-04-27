"""
agent/tools/lab/range_checker.py
----------------------------------
Pure Python module — zero API calls, instant execution.

Loads normal_ranges.json once at module import.
For each extracted RawParameter, finds the matching reference range
and computes status (normal / high / low / critical / borderline).

The matching logic handles common name variants and aliases
so that "Hgb", "Hemoglobin", "haemoglobin" all match the same entry.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path

from agent.tools.lab.schemas.lab_schemas import RawParameter, LabParameter

logger = logging.getLogger(__name__)

# ── Load reference ranges once at import ─────────────────────────────────────
_RANGES_PATH = Path(__file__).resolve().parents[3] / "data" / "lab_reference" / "normal_ranges.json"

try:
    with open(_RANGES_PATH, encoding="utf-8") as f:
        _RANGES: dict = json.load(f)
    logger.info(f"[range_checker] Loaded {len(_RANGES)} reference ranges.")
except FileNotFoundError:
    _RANGES = {}
    logger.error(f"[range_checker] normal_ranges.json not found at {_RANGES_PATH}")


# ── Name normalisation aliases ────────────────────────────────────────────────
# Maps common variants/abbreviations to the canonical key in normal_ranges.json
_ALIASES: dict[str, str] = {
    "hgb": "Hemoglobin",
    "haemoglobin": "Hemoglobin",
    "hb": "Hemoglobin",
    "rbc count": "RBC",
    "red blood cells": "RBC",
    "red blood cell count": "RBC",
    "white blood cell count": "WBC",
    "white blood cells": "WBC",
    "leukocytes": "WBC",
    "thrombocytes": "Platelets",
    "plt": "Platelets",
    "hct": "Hematocrit",
    "packed cell volume": "Hematocrit",
    "pcv": "Hematocrit",
    "blood sugar": "Glucose",
    "fasting glucose": "Glucose",
    "fbs": "Glucose",
    "rbs": "Glucose",
    "glycated hemoglobin": "HbA1c",
    "glycated haemoglobin": "HbA1c",
    "a1c": "HbA1c",
    "hba1c": "HbA1c",
    "hemoglobin a1c": "HbA1c",
    "bad cholesterol": "LDL",
    "ldl cholesterol": "LDL",
    "good cholesterol": "HDL",
    "hdl cholesterol": "HDL",
    "total chol": "TotalCholesterol",
    "cholesterol": "TotalCholesterol",
    "trig": "Triglycerides",
    "tg": "Triglycerides",
    "sgpt": "ALT",
    "alanine aminotransferase": "ALT",
    "sgot": "AST",
    "aspartate aminotransferase": "AST",
    "alkaline phosphatase": "ALP",
    "total bili": "TotalBilirubin",
    "bilirubin": "TotalBilirubin",
    "direct bili": "DirectBilirubin",
    "kidney": "Creatinine",
    "creat": "Creatinine",
    "urea": "BUN",
    "blood urea": "BUN",
    "blood urea nitrogen": "BUN",
    "gfr": "eGFR",
    "egfr": "eGFR",
    "uric acid": "UricAcid",
    "thyroid stimulating hormone": "TSH",
    "thyrotropin": "TSH",
    "free t4": "FreeT4",
    "t4": "FreeT4",
    "free t3": "FreeT3",
    "t3": "FreeT3",
    "iron stores": "Ferritin",
    "serum ferritin": "Ferritin",
    "iron": "SerumIron",
    "b12": "Vitamin_B12",
    "vitamin b12": "Vitamin_B12",
    "cobalamin": "Vitamin_B12",
    "vit d": "VitaminD",
    "vitamin d": "VitaminD",
    "25-oh vitamin d": "VitaminD",
    "c reactive protein": "CRP",
    "c-reactive protein": "CRP",
    "sedimentation rate": "ESR",
    "sed rate": "ESR",
    "k": "Potassium",
    "na": "Sodium",
    "cl": "Chloride",
    "ca": "Calcium",
    "mg": "Magnesium",
    "p": "Phosphorus",
    "psa": "PSA",
    "prostate antigen": "PSA",
    "inr": "INR",
    "prothrombin": "PT",
    "ldh": "LDH",
}


def _normalise_name(name: str) -> str | None:
    """
    Try to match a raw parameter name to a canonical key in _RANGES.
    Returns the canonical key, or None if no match found.
    """
    # Try exact match first
    if name in _RANGES:
        return name

    # Try case-insensitive exact match
    name_lower = name.lower().strip()
    for key in _RANGES:
        if key.lower() == name_lower:
            return key

    # Try alias lookup
    alias_key = _ALIASES.get(name_lower)
    if alias_key and alias_key in _RANGES:
        return alias_key

    # Try partial match (name is a substring of a key or vice versa)
    for key in _RANGES:
        if name_lower in key.lower() or key.lower() in name_lower:
            return key

    return None


def _compute_status(
    value: float,
    ref: dict,
) -> tuple[str, float]:
    """
    Determine status and deviation percentage for a parameter value.

    Returns:
        (status_string, deviation_percent)
        deviation_percent: positive = above max, negative = below min
    """
    normal_min = ref.get("normal_min", 0.0)
    normal_max = ref.get("normal_max", 999.0)
    critical_low = ref.get("critical_low")
    critical_high = ref.get("critical_high")
    borderline_high = ref.get("borderline_high")

    # Critical checks first (highest priority)
    if critical_low is not None and value < critical_low:
        deviation = ((critical_low - value) / critical_low * 100) if critical_low > 0 else 0
        return "critical_low", -round(deviation, 1)

    if critical_high is not None and value > critical_high:
        deviation = ((value - critical_high) / critical_high * 100) if critical_high > 0 else 0
        return "critical_high", round(deviation, 1)

    # Normal range check
    if value < normal_min:
        deviation = ((normal_min - value) / normal_min * 100) if normal_min > 0 else 0
        return "low", -round(deviation, 1)

    if borderline_high is not None and normal_max < value <= borderline_high:
        deviation = ((value - normal_max) / normal_max * 100) if normal_max > 0 else 0
        return "borderline_high", round(deviation, 1)

    if value > normal_max:
        deviation = ((value - normal_max) / normal_max * 100) if normal_max > 0 else 0
        return "high", round(deviation, 1)

    return "normal", 0.0


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def check_ranges(raw_params: list[RawParameter]) -> list[LabParameter]:
    """
    For each extracted parameter, look up the normal range and compute status.

    Args:
        raw_params: List of RawParameter objects from parameter_extractor.py

    Returns:
        List of LabParameter objects enriched with range, status, deviation.
        Parameters not found in the reference DB get status="unknown".
    """
    checked: list[LabParameter] = []

    for raw in raw_params:
        canonical_name = _normalise_name(raw.name)

        if canonical_name is None or canonical_name not in _RANGES:
            # Parameter not in our reference DB — include it as unknown
            logger.debug(f"[range_checker] '{raw.name}' not found in reference DB.")
            checked.append(LabParameter(
                name=raw.name,
                full_name=raw.name,
                value=raw.value,
                unit=raw.unit,
                normal_min=0.0,
                normal_max=0.0,
                status="unknown",
                deviation_percent=0.0,
                panel="",
                clinical_note="Reference range not found in database.",
            ))
            continue

        ref = _RANGES[canonical_name]
        status, deviation = _compute_status(raw.value, ref)

        checked.append(LabParameter(
            name=canonical_name,
            full_name=ref.get("full_name", canonical_name),
            value=raw.value,
            unit=raw.unit or ref.get("unit", ""),
            normal_min=ref.get("normal_min", 0.0),
            normal_max=ref.get("normal_max", 0.0),
            status=status,
            deviation_percent=deviation,
            panel=ref.get("panel", ""),
            clinical_note=ref.get("clinical_note", ""),
        ))

    logger.info(f"[range_checker] Checked {len(checked)} parameters.")
    return checked