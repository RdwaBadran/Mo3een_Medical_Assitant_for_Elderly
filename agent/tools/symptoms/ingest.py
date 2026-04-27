"""
agent/tools/symptoms/ingest.py
--------------------------------
Multi-source ingestion orchestrator for the elderly-disease knowledge base.

Data sources
------------
  1. PubMed      — NCBI scientific abstracts (~1,750 docs via 35 queries)
  2. Wikipedia   — Full disease articles      (~46 articles)
  3. MedlinePlus — NIH NLM encyclopedia pages (~35 pages)
  4. WHO         — Official fact sheets        (~20 pages)
  5. CDC         — US health topic pages       (~20 pages)
  6. Local files — Any .txt placed in data/medical_docs/ (always included)

Usage
-----
Run from project root:
    python -m agent.tools.symptoms.ingest
    python -m agent.tools.symptoms.ingest --skip-pubmed

The script saves every fetched document as a .txt file under
data/medical_docs/ so you have a local cache and can re-index
without re-downloading.
"""

import argparse
import sys
import time
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ── Paths ─────────────────────────────────────────────────────────────────
# parents[3] = agent/tools/symptoms/ingest.py → go up 3 levels to project root
ROOT       = Path(__file__).resolve().parents[3]   # F:\medical-agent\
DOCS_DIR   = ROOT / "data" / "medical_docs"
CHROMA_DIR = ROOT / "data" / "chroma_db"
COLLECTION = "symptoms_kb"                          # must match rag.py

# ── Embedding model ────────────────────────────────────────────────────────
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # must match rag.py

# ── Chunking config ────────────────────────────────────────────────────────
CHUNK_SIZE    = 600
CHUNK_OVERLAP = 80


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _save_doc(fname: str, text: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / fname).write_text(text, encoding="utf-8")


def _load_local_docs() -> list[Document]:
    if not DOCS_DIR.exists() or not any(DOCS_DIR.glob("**/*.txt")):
        return []
    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    return loader.load()


# ══════════════════════════════════════════════════════════════════════════
# Source runners — each imports from agent/tools/symptoms/sources/
# ══════════════════════════════════════════════════════════════════════════

def _run_pubmed(max_per_query: int = 50) -> int:
    from agent.tools.symptoms.sources import pubmed
    count = 0
    for fname, text in pubmed.fetch_all(max_per_query=max_per_query):
        _save_doc(fname, text)
        count += 1
    return count


def _run_wikipedia() -> int:
    from agent.tools.symptoms.sources import wikipedia_src
    count = 0
    for fname, text in wikipedia_src.fetch_all():
        _save_doc(fname, text)
        count += 1
    return count


def _run_medlineplus() -> int:
    from agent.tools.symptoms.sources import medlineplus
    count = 0
    for fname, text in medlineplus.fetch_all():
        _save_doc(fname, text)
        count += 1
    return count


def _run_who() -> int:
    from agent.tools.symptoms.sources import who
    count = 0
    for fname, text in who.fetch_all():
        _save_doc(fname, text)
        count += 1
    return count


def _run_cdc() -> int:
    from agent.tools.symptoms.sources import cdc
    count = 0
    for fname, text in cdc.fetch_all():
        _save_doc(fname, text)
        count += 1
    return count


SOURCE_MAP = {
    "pubmed":      _run_pubmed,
    "wikipedia":   _run_wikipedia,
    "medlineplus": _run_medlineplus,
    "who":         _run_who,
    "cdc":         _run_cdc,
}


# ══════════════════════════════════════════════════════════════════════════
# Main ingestion pipeline
# ══════════════════════════════════════════════════════════════════════════

def ingest(sources: list[str] | None = None) -> int:
    active = sources or list(SOURCE_MAP.keys())
    t0 = time.time()

    # Step 1: Download & save
    total_docs = 0
    for name in active:
        if name not in SOURCE_MAP:
            print(f"[ingest] Unknown source {name!r} — skipping.")
            continue
        print(f"\n[ingest] ── Source: {name.upper()} ──")
        try:
            n = SOURCE_MAP[name]()
            print(f"[ingest]    ✓ Saved {n} documents from {name}.")
            total_docs += n
        except Exception as exc:
            print(f"[ingest]    ✗ {name} failed: {exc}")

    print(f"\n[ingest] Downloaded {total_docs} new documents in {time.time() - t0:.1f}s.")

    # Step 2: Load all local docs
    print("\n[ingest] Loading all documents from disk …")
    raw_docs = _load_local_docs()
    if not raw_docs:
        print("[ingest] No documents found. Add .txt files to data/medical_docs/ and re-run.")
        return 0
    print(f"[ingest] Loaded {len(raw_docs)} documents.")

    # Step 3: Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"[ingest] Split into {len(chunks)} chunks.")

    # Step 4: Embed & store
    print("[ingest] Loading embedding model (first run downloads ~90 MB) …")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[ingest] Storing vectors in ChromaDB at {CHROMA_DIR} …")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION,
    )

    elapsed = time.time() - t0
    print(f"\n[ingest] ✅ Done — {len(chunks)} chunks stored in {elapsed:.1f}s ({elapsed/60:.1f} min).")
    return len(chunks)


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest medical knowledge into ChromaDB.")
    parser.add_argument(
        "--sources", nargs="+",
        choices=list(SOURCE_MAP.keys()),
        default=list(SOURCE_MAP.keys()),
    )
    parser.add_argument("--skip-pubmed", action="store_true")
    args = parser.parse_args()

    selected = args.sources
    if args.skip_pubmed and "pubmed" in selected:
        selected = [s for s in selected if s != "pubmed"]
        print("[ingest] Skipping PubMed.")

    n = ingest(sources=selected)
    sys.exit(0 if n > 0 else 1)