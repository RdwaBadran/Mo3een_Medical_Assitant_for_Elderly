"""
agent/tools/symptoms/sources/wikipedia_src.py
------------------------------------------------
Fetches full Wikipedia articles for ~45 elderly-disease topics using the
Wikipedia REST API (no authentication required).

Each article is returned as plain text (section headers preserved) so the
chunker in ingest.py can split it meaningfully.
"""

import time
from typing import Iterator

import requests

API   = "https://en.wikipedia.org/w/api.php"
DELAY = 0.5   # seconds between requests — Wikipedia's guidance

HEADERS = {
    "User-Agent": "MedicalAgentBot/1.0 (elderly-disease knowledge base; educational)"
}

# ── Topics (elderly-focused) ───────────────────────────────────────────────
WIKIPEDIA_TOPICS: list[str] = [
    "Alzheimer's disease",
    "Parkinson's disease",
    "Dementia",
    "Lewy body dementia",
    "Vascular dementia",
    "Osteoporosis",
    "Osteoarthritis",
    "Rheumatoid arthritis",
    "Gout",
    "Heart failure",
    "Coronary artery disease",
    "Atrial fibrillation",
    "Hypertension",
    "Stroke",
    "Transient ischemic attack",
    "Peripheral artery disease",
    "Chronic obstructive pulmonary disease",
    "Pneumonia",
    "Influenza",
    "Type 2 diabetes",
    "Chronic kidney disease",
    "Glaucoma",
    "Cataract",
    "Age-related macular degeneration",
    "Hearing loss",
    "Urinary incontinence",
    "Benign prostatic hyperplasia",
    "Depression (mood)",
    "Anxiety disorder",
    "Insomnia",
    "Delirium",
    "Herpes zoster",
    "Sarcopenia",
    "Frailty syndrome",
    "Malnutrition",
    "Pressure ulcer",
    "Falls in older adults",
    "Polypharmacy",
    "Hypothyroidism",
    "Anemia",
    "Colorectal cancer",
    "Prostate cancer",
    "Breast cancer",
    "Lung cancer",
    "Geriatrics",
    "Palliative care",
]


def _fetch_article(title: str) -> str:
    """Return full plain-text content of a Wikipedia article."""
    params = {
        "action":          "query",
        "prop":            "extracts",
        "titles":          title,
        "explaintext":     "1",
        "exsectionformat": "plain",
        "format":          "json",
        "redirects":       "1",
    }
    resp = requests.get(API, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    page  = next(iter(pages.values()))
    return page.get("extract", "")


def fetch_all() -> Iterator[tuple[str, str]]:
    """
    Yields (filename, text_content) pairs for every Wikipedia topic.

    Yields:
        tuple[str, str]: (safe_filename, full_article_text)
    """
    for topic in WIKIPEDIA_TOPICS:
        print(f"    [Wikipedia] ↓ {topic!r}")
        try:
            text = _fetch_article(topic)
            if len(text) > 300:   # skip redirect stubs
                safe  = topic.replace(" ", "_").replace("/", "-").replace("'", "")
                fname = f"wikipedia_{safe}.txt"
                yield fname, f"# {topic}\n\n{text}"
            time.sleep(DELAY)
        except Exception as exc:
            print(f"    [Wikipedia] ✗ {topic!r}: {exc}")
            continue
