from __future__ import annotations

"""
chroma_client.py
----------------
Thread-safe singleton wrapper around a ChromaDB PersistentClient.

Usage:
    from db.chroma_client import get_client, get_collection

Environment variables:
    CHROMA_DB_PATH  - filesystem path where ChromaDB persists data
                      (default: data/chroma_db)
"""

import os
import threading
import chromadb
from chromadb.config import Settings

# ---------------------------------------------------------------------------
# Collection name constants
# ---------------------------------------------------------------------------
COLLECTION_CORPUS   = "document_corpus"
COLLECTION_HISTORY  = "analysis_history"
COLLECTION_PROFILES = "document_profiles"

_VALID_COLLECTIONS = {COLLECTION_CORPUS, COLLECTION_HISTORY, COLLECTION_PROFILES}

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------
_client: chromadb.PersistentClient | None = None
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_client() -> chromadb.PersistentClient:
    """Return the shared ChromaDB client, creating it on first call."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:  # double-checked locking
                db_path = os.environ.get("CHROMA_DB_PATH", "data/chroma_db")
                os.makedirs(db_path, exist_ok=True)
                _client = chromadb.PersistentClient(
                    path=db_path,
                    settings=Settings(anonymized_telemetry=False),
                )
                _initialize_collections(_client)
    return _client


def get_collection(name: str) -> chromadb.Collection:
    """Return a named collection from the singleton client.

    Raises:
        ValueError: if *name* is not one of the recognised collection names.
    """
    if name not in _VALID_COLLECTIONS:
        raise ValueError(
            f"Unknown collection '{name}'. "
            f"Valid options: {sorted(_VALID_COLLECTIONS)}"
        )
    return get_client().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_client() -> None:
    """Null out the singleton.  Intended for use in test teardown only."""
    global _client
    with _lock:
        _client = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _initialize_collections(client: chromadb.PersistentClient) -> None:
    """Ensure all required collections exist with consistent metadata."""
    for name in _VALID_COLLECTIONS:
        client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
