"""
agent/tools/symptoms/sources/cdc.py
-------------------------------------
Scrapes CDC (Centers for Disease Control and Prevention) health topic
pages for elderly-relevant conditions.

CDC pages are publicly accessible US government content with no
authentication required.

Coverage: ~20 elderly-focused health topic pages.
"""

import time
from typing import Iterator

import requests
from bs4 import BeautifulSoup

DELAY = 0.8

HEADERS = {
    "User-Agent": "MedicalAgentBot/1.0 (elderly-disease knowledge base; educational)",
    "Accept-Language": "en-US,en;q=0.9",
}

# ── Topic → URL map ────────────────────────────────────────────────────────
TOPICS: dict[str, str] = {
    "alzheimers":          "https://www.cdc.gov/alzheimers/index.html",
    "arthritis_elderly":   "https://www.cdc.gov/arthritis/index.html",
    "cancer_prevention":   "https://www.cdc.gov/cancer/index.htm",
    "copd":                "https://www.cdc.gov/copd/index.html",
    "diabetes":            "https://www.cdc.gov/diabetes/index.html",
    "falls_prevention":    "https://www.cdc.gov/falls/index.html",
    "heart_disease":       "https://www.cdc.gov/heartdisease/index.htm",
    "high_blood_pressure": "https://www.cdc.gov/bloodpressure/index.htm",
    "influenza_elderly":   "https://www.cdc.gov/flu/highrisk/65over.htm",
    "kidney_disease":      "https://www.cdc.gov/kidneydisease/index.html",
    "obesity_elderly":     "https://www.cdc.gov/obesity/index.html",
    "osteoporosis":        "https://www.cdc.gov/bonehealthforlife/index.html",
    "pneumonia":           "https://www.cdc.gov/pneumonia/index.html",
    "shingles_elderly":    "https://www.cdc.gov/shingles/index.html",
    "smoking_elderly":     "https://www.cdc.gov/tobacco/campaign/tips/index.html",
    "stroke":              "https://www.cdc.gov/stroke/index.htm",
    "vision_health":       "https://www.cdc.gov/visionhealth/index.htm",
    "healthy_ageing":      "https://www.cdc.gov/aging/index.html",
    "depression_elderly":  "https://www.cdc.gov/mentalhealth/learn/index.htm",
    "physical_activity":   "https://www.cdc.gov/physicalactivity/index.html",
}


def _scrape(url: str) -> str:
    """Download and return clean text from a CDC health topic page."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # Remove noise
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "form", "noscript", "button"]):
        tag.decompose()

    # CDC uses 'syndicate' div or 'main' for body content
    main = (
        soup.find("div", class_="syndicate")
        or soup.find("main")
        or soup.find("div", id="content")
        or soup.body
    )
    return main.get_text(separator="\n", strip=True) if main else ""


def fetch_all() -> Iterator[tuple[str, str]]:
    """
    Yields (filename, text_content) pairs for every CDC health topic.

    Yields:
        tuple[str, str]: (safe_filename, scraped_text)
    """
    for name, url in TOPICS.items():
        print(f"    [CDC] ↓ {name!r}")
        try:
            text = _scrape(url)
            if len(text) > 200:
                yield f"cdc_{name}.txt", text
            time.sleep(DELAY)
        except Exception as exc:
            print(f"    [CDC] ✗ {name!r}: {exc}")
            continue
