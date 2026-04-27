"""
agent/tools/symptoms/sources/medlineplus.py
---------------------------------------------
Scrapes the NIH MedlinePlus medical encyclopedia — the US National
Library of Medicine's consumer-health resource.  Plain English, reliable,
and freely accessible without authentication.

Coverage: ~35 elderly-relevant topic pages.
"""

import time
from typing import Iterator

import requests
from bs4 import BeautifulSoup

DELAY = 0.8   # be polite — government server

HEADERS = {
    "User-Agent": "MedicalAgentBot/1.0 (elderly-disease knowledge base; educational)"
}

# ── Topic → URL map ────────────────────────────────────────────────────────
TOPICS: dict[str, str] = {
    "alzheimers_disease":        "https://medlineplus.gov/alzheimersdisease.html",
    "parkinsons_disease":        "https://medlineplus.gov/parkinsonsdisease.html",
    "dementia":                  "https://medlineplus.gov/dementia.html",
    "osteoporosis":              "https://medlineplus.gov/osteoporosis.html",
    "osteoarthritis":            "https://medlineplus.gov/osteoarthritis.html",
    "rheumatoid_arthritis":      "https://medlineplus.gov/rheumatoidarthritis.html",
    "heart_failure":             "https://medlineplus.gov/heartfailure.html",
    "coronary_artery_disease":   "https://medlineplus.gov/coronaryarterydisease.html",
    "atrial_fibrillation":       "https://medlineplus.gov/atrialfibrillation.html",
    "high_blood_pressure":       "https://medlineplus.gov/highbloodpressure.html",
    "stroke":                    "https://medlineplus.gov/stroke.html",
    "peripheral_artery_disease": "https://medlineplus.gov/peripheralarterialdisease.html",
    "copd":                      "https://medlineplus.gov/copd.html",
    "pneumonia":                 "https://medlineplus.gov/pneumonia.html",
    "flu":                       "https://medlineplus.gov/flu.html",
    "diabetes_type2":            "https://medlineplus.gov/diabetestype2.html",
    "chronic_kidney_disease":    "https://medlineplus.gov/kidneydiseases.html",
    "glaucoma":                  "https://medlineplus.gov/glaucoma.html",
    "cataracts":                 "https://medlineplus.gov/cataract.html",
    "macular_degeneration":      "https://medlineplus.gov/maculardegeneration.html",
    "hearing_loss":              "https://medlineplus.gov/hearingdisordersanddeafness.html",
    "urinary_incontinence":      "https://medlineplus.gov/urinaryincontinence.html",
    "enlarged_prostate":         "https://medlineplus.gov/enlargedprostate.html",
    "depression":                "https://medlineplus.gov/depression.html",
    "sleep_disorders":           "https://medlineplus.gov/sleepdisorders.html",
    "shingles":                  "https://medlineplus.gov/shingles.html",
    "falls":                     "https://medlineplus.gov/falls.html",
    "malnutrition":              "https://medlineplus.gov/malnutrition.html",
    "pressure_sores":            "https://medlineplus.gov/pressuresores.html",
    "colorectal_cancer":         "https://medlineplus.gov/colorectalcancer.html",
    "prostate_cancer":           "https://medlineplus.gov/prostatecancer.html",
    "breast_cancer":             "https://medlineplus.gov/breastcancer.html",
    "hypothyroidism":            "https://medlineplus.gov/hypothyroidism.html",
    "anemia":                    "https://medlineplus.gov/anemia.html",
    "delirium":                  "https://medlineplus.gov/delirium.html",
}


def _scrape(url: str) -> str:
    """Download and return clean text from a MedlinePlus topic page."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # Remove noise
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "form", "noscript"]):
        tag.decompose()

    # Prefer the main content area
    main = (
        soup.find("div", id="topic-summary")
        or soup.find("main")
        or soup.find("article")
        or soup.body
    )
    return main.get_text(separator="\n", strip=True) if main else ""


def fetch_all() -> Iterator[tuple[str, str]]:
    """
    Yields (filename, text_content) pairs for every MedlinePlus topic.

    Yields:
        tuple[str, str]: (safe_filename, scraped_text)
    """
    for name, url in TOPICS.items():
        print(f"    [MedlinePlus] ↓ {name!r}")
        try:
            text = _scrape(url)
            if len(text) > 200:
                yield f"medlineplus_{name}.txt", text
            time.sleep(DELAY)
        except Exception as exc:
            print(f"    [MedlinePlus] ✗ {name!r}: {exc}")
            continue
