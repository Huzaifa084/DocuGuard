from __future__ import annotations

"""
corpus_store.py
---------------
CRUD operations for the document corpus collection.

Each document is split into overlapping text chunks before storage so that
semantic search can surface relevant passages rather than whole documents.

Constants
---------
CHUNK_WORD_SIZE : int
    Target number of words per chunk.
CHUNK_STRIDE : int
    Number of words to advance the window between consecutive chunks
    (overlap = CHUNK_WORD_SIZE - CHUNK_STRIDE words).
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from db.chroma_client import get_collection, COLLECTION_CORPUS

# ---------------------------------------------------------------------------
# Chunking parameters
# ---------------------------------------------------------------------------
CHUNK_WORD_SIZE: int = 400
CHUNK_STRIDE: int = 50


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_document(
    text: str,
    filename: str,
    extra_metadata: dict[str, Any] | None = None,
) -> str:
    """Chunk *text* and store all chunks in the corpus collection.

    Parameters
    ----------
    text:
        Full document text.
    filename:
        Original filename, stored in each chunk's metadata.
    extra_metadata:
        Any additional key-value pairs to attach to every chunk.

    Returns
    -------
    str
        The newly assigned document ID (UUID4 hex string).
    """
    doc_id = uuid.uuid4().hex
    upload_date = datetime.now(timezone.utc).isoformat()
    chunks = _chunk_text(text)

    collection = get_collection(COLLECTION_CORPUS)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{idx}"
        meta: dict[str, Any] = {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": idx,
            "total_chunks": len(chunks),
            "upload_date": upload_date,
        }
        if extra_metadata:
            meta.update(extra_metadata)

        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append(meta)

    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

    return doc_id


def query_similar(
    query_text: str,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    """Return the *n_results* most semantically similar chunks to *query_text*.

    Parameters
    ----------
    query_text:
        The text to search for.
    n_results:
        Maximum number of results to return.

    Returns
    -------
    list of dict
        Each dict contains keys: ``id``, ``document``, ``distance``,
        and all metadata fields stored on the chunk.
    """
    collection = get_collection(COLLECTION_CORPUS)

    # Guard: ChromaDB raises if n_results > actual document count.
    actual_count = collection.count()
    if actual_count == 0:
        return []

    clamped = min(n_results, actual_count)

    raw = collection.query(
        query_texts=[query_text],
        n_results=clamped,
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB returns nested lists (one list per query); flatten for single query.
    results: list[dict[str, Any]] = []
    for chunk_id, doc, meta, dist in zip(
        raw["ids"][0],
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        results.append(
            {
                "id": chunk_id,
                "document": doc,
                "distance": dist,
                **meta,
            }
        )

    return results


def list_documents() -> list[dict[str, Any]]:
    """Return one summary record per document, sorted newest-first.

    Deduplicates by ``doc_id`` so that multi-chunk documents appear only once.
    Each record contains: ``doc_id``, ``filename``, ``upload_date``,
    ``total_chunks``, and any extra metadata stored on the first chunk.

    Returns
    -------
    list of dict
    """
    collection = get_collection(COLLECTION_CORPUS)
    if collection.count() == 0:
        return []

    # Fetch all metadatas (no embeddings needed).
    raw = collection.get(include=["metadatas"])

    seen: dict[str, dict[str, Any]] = {}
    for meta in raw["metadatas"]:
        doc_id: str = meta["doc_id"]
        if doc_id not in seen:
            seen[doc_id] = dict(meta)

    docs = list(seen.values())
    docs.sort(key=lambda d: d.get("upload_date", ""), reverse=True)
    return docs


def delete_document(doc_id: str) -> bool:
    """Delete all chunks belonging to *doc_id*.

    Returns
    -------
    bool
        ``True`` if at least one chunk was deleted, ``False`` if the document
        was not found.
    """
    collection = get_collection(COLLECTION_CORPUS)
    if collection.count() == 0:
        return False

    raw = collection.get(include=["metadatas"])

    chunk_ids = [
        chunk_id
        for chunk_id, meta in zip(raw["ids"], raw["metadatas"])
        if meta.get("doc_id") == doc_id
    ]

    if not chunk_ids:
        return False

    collection.delete(ids=chunk_ids)
    return True


def get_all_texts() -> list[tuple[str, str]]:
    """Return ``(chunk_text, filename)`` for every chunk in the corpus.

    Intended for TF-IDF or other corpus-wide analysis that needs raw text.

    Returns
    -------
    list of (str, str)
    """
    collection = get_collection(COLLECTION_CORPUS)
    if collection.count() == 0:
        return []

    raw = collection.get(include=["documents", "metadatas"])
    return [
        (doc, meta.get("filename", ""))
        for doc, meta in zip(raw["documents"], raw["metadatas"])
    ]


def corpus_count() -> int:
    """Return the total number of chunks stored in the corpus collection."""
    return get_collection(COLLECTION_CORPUS).count()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chunk_text(text: str) -> list[str]:
    """Split *text* into overlapping word-window chunks.

    Parameters
    ----------
    text:
        The full document text to split.

    Returns
    -------
    list of str
        Non-empty chunk strings.  Returns ``[text]`` if the text is shorter
        than one full chunk.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + CHUNK_WORD_SIZE
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += CHUNK_STRIDE

    return chunks
