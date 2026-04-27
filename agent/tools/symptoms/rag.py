"""
agent/tools/symptoms/rag.py
----------------------------
ChromaDB retrieval helper for the symptoms tool.

Uses sentence-transformers (local, free) for embeddings — same model
used in ingest.py, so the vectors are compatible.

Public function: retrieve_context(query: str) -> str
Returns the top-k relevant chunks as a single newline-separated string,
or an empty string if the DB is empty / not yet built.
"""

from __future__ import annotations
import logging
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

# ── Paths & config ─────────────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parents[3]   # F:\medical-agent\
CHROMA_DIR      = ROOT / "data" / "chroma_db"
COLLECTION_NAME = "symptoms_kb"                          # must match ingest.py
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"  # must match ingest.py
TOP_K           = 5

# ── Lazy-loaded vectorstore singleton ──────────────────────────────────────
_vectorstore: Chroma | None = None


def _get_vectorstore() -> Chroma | None:
    """
    Load (or return cached) ChromaDB vector store.
    Returns None if the DB directory doesn't exist yet (ingest not run).
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    if not CHROMA_DIR.exists():
        logger.warning(
            f"[rag] ChromaDB not found at {CHROMA_DIR}. "
            "Run: python -m agent.tools.symptoms.ingest"
        )
        return None

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        count = _vectorstore._collection.count()
        logger.info(f"[rag] ChromaDB loaded — {count} chunks available.")
        return _vectorstore
    except Exception as exc:
        logger.error(f"[rag] Failed to load ChromaDB: {exc}")
        return None


def retrieve_context(query: str) -> str:
    """
    Retrieve the most relevant document chunks for a symptoms query.

    Args:
        query: The patient's symptom description string.

    Returns:
        Newline-separated string of top-k relevant chunks,
        or empty string if no documents are indexed.
    """
    vectorstore = _get_vectorstore()
    if vectorstore is None:
        return ""

    try:
        docs = vectorstore.similarity_search(query, k=TOP_K)
        if not docs:
            return ""
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception as exc:
        logger.error(f"[rag] Retrieval failed: {exc}")
        return ""