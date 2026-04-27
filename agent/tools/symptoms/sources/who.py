"""
agent/tools/symptoms/sources/who.py
-------------------------------------
Scrapes WHO (World Health Organization) fact sheets relevant to elderly
health.  WHO fact sheets are publicly accessible, authoritative, and
frequently updated.

Coverage: ~20 fact sheets on the most impactful elderly conditions.
"""

import time
from typing import Iterator

import requests
from bs4 import BeautifulSoup

DELAY = 1.0   # WHO servers — be extra polite

HEADERS = {
    "User-Agent": "MedicalAgentBot/1.0 (elderly-disease knowledge base; educational)",
    "Accept-Language": "en-US,en;q=0.9",
}

# ── Fact sheet URLs ────────────────────────────────────────────────────────
FACTSHEETS: dict[str, str] = {
    "ageing_and_health":      "https://www.who.int/news-room/fact-sheets/detail/ageing-and-health",
    "dementia":               "https://www.who.int/news-room/fact-sheets/detail/dementia",
    "falls":                  "https://www.who.int/news-room/fact-sheets/detail/falls",
    "cardiovascular_disease": "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)",
    "diabetes":               "https://www.who.int/news-room/fact-sheets/detail/diabetes",
    "hypertension":           "https://www.who.int/news-room/fact-sheets/detail/hypertension",
    "cancer":                 "https://www.who.int/news-room/fact-sheets/detail/cancer",
    "copd":                   "https://www.who.int/news-room/fact-sheets/detail/chronic-obstructive-pulmonary-disease-(copd)",
    "depression":             "https://www.who.int/news-room/fact-sheets/detail/depression",
    "parkinson_disease":      "https://www.who.int/news-room/fact-sheets/detail/parkinson-disease",
    "osteoarthritis":         "https://www.who.int/news-room/fact-sheets/detail/osteoarthritis",
    "visual_impairment":      "https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment",
    "hearing_loss":           "https://www.who.int/news-room/fact-sheets/detail/deafness-and-hearing-loss",
    "malnutrition":           "https://www.who.int/news-room/fact-sheets/detail/malnutrition",
    "pneumonia":              "https://www.who.int/news-room/fact-sheets/detail/pneumonia",
    "influenza":              "https://www.who.int/news-room/fact-sheets/detail/influenza-(seasonal)",
    "stroke":                 "https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death",
    "palliative_care":        "https://www.who.int/news-room/fact-sheets/detail/palliative-care",
    "tobacco_elderly":        "https://www.who.int/news-room/fact-sheets/detail/tobacco",
    "physical_activity":      "https://www.who.int/news-room/fact-sheets/detail/physical-activity",
}


def _scrape(url: str) -> str:
    """Download and return clean text from a WHO fact-sheet page."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "form", "noscript", "button", "figure"]):
        tag.decompose()

    # WHO article body lives in a specific container
    article = (
        soup.find("div", class_="sf-detail-body-wrapper")
        or soup.find("div", {"id": "PageContent"})
        or soup.find("article")
        or soup.find("main")
        or soup.body
    )
    return article.get_text(separator="\n", strip=True) if article else ""


def fetch_all() -> Iterator[tuple[str, str]]:
    """
    Yields (filename, text_content) pairs for every WHO fact sheet.

    Yields:
        tuple[str, str]: (safe_filename, scraped_text)
    """
    for name, url in FACTSHEETS.items():
        print(f"    [WHO] ↓ {name!r}")
        try:
            text = _scrape(url)
            if len(text) > 200:
                yield f"who_{name}.txt", text
            time.sleep(DELAY)
        except Exception as exc:
            print(f"    [WHO] ✗ {name!r}: {exc}")
            continue
