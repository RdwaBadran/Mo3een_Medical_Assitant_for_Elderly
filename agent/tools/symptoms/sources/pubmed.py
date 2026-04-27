"""
agent/tools/symptoms/sources/pubmed.py
----------------------------------------
Fetches medical abstracts from PubMed via the free NCBI E-utilities API.
No API key required (respects the 3 req/sec limit automatically).
With a free NCBI API key the limit rises to 10 req/sec — add it to .env
as  NCBI_API_KEY=<your_key>  to enable faster downloads.

Coverage: ~35 elderly-disease queries × up to 50 abstracts = ~1,750 documents.
"""

import os
import time
import xml.etree.ElementTree as ET
from typing import Iterator

import requests
from dotenv import load_dotenv

load_dotenv()

# ── API endpoints ──────────────────────────────────────────────────────────
ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Polite rate-limiting (0.4 s → ≤ 3 req/sec without key, 0.15 s with key)
_API_KEY = os.getenv("NCBI_API_KEY", "")
DELAY    = 0.15 if _API_KEY else 0.4
BATCH    = 100    # PMIDs per efetch call

# ── Disease queries (elderly-specific) ────────────────────────────────────
ELDERLY_QUERIES: list[str] = [
    "Alzheimer disease elderly diagnosis treatment",
    "Parkinson disease older adults management",
    "dementia elderly care cognitive decline",
    "osteoporosis elderly fracture prevention",
    "osteoarthritis knee hip elderly treatment",
    "heart failure older adults prognosis",
    "atrial fibrillation elderly anticoagulation",
    "hypertension elderly management guidelines",
    "stroke prevention elderly risk factors",
    "COPD elderly treatment outcomes",
    "type 2 diabetes older adults management",
    "chronic kidney disease elderly",
    "glaucoma elderly diagnosis treatment",
    "age-related macular degeneration AMD treatment",
    "cataract elderly surgery outcomes",
    "urinary incontinence elderly women men",
    "benign prostatic hyperplasia elderly treatment",
    "depression elderly antidepressants",
    "anxiety disorder older adults treatment",
    "insomnia sleep disorders elderly",
    "peripheral artery disease elderly claudication",
    "herpes zoster shingles elderly vaccination",
    "pneumonia elderly community-acquired outcomes",
    "sarcopenia muscle loss elderly prevention",
    "frailty syndrome elderly assessment",
    "polypharmacy elderly adverse drug events",
    "malnutrition elderly nutritional assessment",
    "pressure ulcers bedsores elderly prevention",
    "delirium elderly hospital management",
    "colorectal cancer elderly screening",
    "prostate cancer elderly watchful waiting",
    "hypothyroidism elderly subclinical treatment",
    "anemia elderly causes treatment",
    "falls elderly risk prevention interventions",
    "geriatric comprehensive geriatric assessment",
]


def _search_pmids(query: str, max_results: int) -> list[str]:
    params: dict = {
        "db":      "pubmed",
        "term":    query,
        "retmax":  max_results,
        "retmode": "json",
    }
    if _API_KEY:
        params["api_key"] = _API_KEY
    resp = requests.get(ESEARCH, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["esearchresult"]["idlist"]


def _fetch_abstracts(pmids: list[str]) -> list[dict]:
    """POST a batch of PMIDs and return list of {title, abstract} dicts."""
    if not pmids:
        return []
    data: dict = {
        "db":      "pubmed",
        "id":      ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    if _API_KEY:
        data["api_key"] = _API_KEY
    resp = requests.post(EFETCH, data=data, timeout=60)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    results: list[dict] = []
    for article in root.iter("PubmedArticle"):
        title_el    = article.find(".//ArticleTitle")
        abstract_el = article.find(".//AbstractText")
        title    = "".join(title_el.itertext())    if title_el    is not None else ""
        abstract = "".join(abstract_el.itertext()) if abstract_el is not None else ""
        if abstract.strip():
            results.append({"title": title.strip(), "text": abstract.strip()})
    return results


def fetch_all(max_per_query: int = 50) -> Iterator[tuple[str, str]]:
    """
    Yields (filename, text_content) pairs for every retrieved abstract.

    Args:
        max_per_query: Maximum number of abstracts to fetch per search query.

    Yields:
        tuple[str, str]: (safe_filename, "Title: ...\n\nAbstract:\n...")
    """
    for query in ELDERLY_QUERIES:
        print(f"    [PubMed] ↓ {query!r}")
        try:
            pmids = _search_pmids(query, max_per_query)
            time.sleep(DELAY)

            for i in range(0, len(pmids), BATCH):
                batch    = pmids[i : i + BATCH]
                articles = _fetch_abstracts(batch)
                time.sleep(DELAY)

                for art in articles:
                    safe = art["title"][:60].replace("/", "-").replace(":", "").strip()
                    fname = f"pubmed_{safe}.txt"
                    text  = f"Title: {art['title']}\n\nAbstract:\n{art['text']}"
                    yield fname, text

        except Exception as exc:
            print(f"    [PubMed] ✗ Error for {query!r}: {exc}")
            continue
