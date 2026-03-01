"""
history_store.py
----------------
Persistence layer for document analysis results.

Each analysis record is stored as a ChromaDB document whose *id* is derived
deterministically from the report ID so that lookups are O(1).
"""

from datetime import datetime, timezone
from typing import Any

from db.chroma_client import get_collection, COLLECTION_HISTORY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record_id(report_id: str) -> str:
    """Return the ChromaDB document ID for a given report ID."""
    return f"analysis_{report_id}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_analysis(
    report_id: str,
    filename: str,
    plagiarism_score: float,
    ai_score: float,
    verdict: str,
    matched_sources: list[str] | None = None,
    fingerprint_similarity: float | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> str:
    """Persist an analysis result.

    Parameters
    ----------
    report_id:
        Unique identifier for this analysis run (caller-supplied).
    filename:
        Original document filename.
    plagiarism_score:
        Plagiarism likelihood in the range [0.0, 1.0].
    ai_score:
        AI-generated content likelihood in the range [0.0, 1.0].
    verdict:
        Human-readable verdict string (e.g. ``"likely_ai"``, ``"clean"``).
    matched_sources:
        Optional list of filenames / URLs that were flagged as similar.
    fingerprint_similarity:
        Optional raw cosine-distance similarity score from fingerprint
        comparison.  Only stored when provided.
    extra_metadata:
        Any additional key-value pairs to attach to the record.

    Returns
    -------
    str
        The ChromaDB document ID used to store the record.
    """
    collection = get_collection(COLLECTION_HISTORY)
    upload_date = datetime.now(timezone.utc).isoformat()

    metadata: dict[str, Any] = {
        "report_id": report_id,
        "filename": filename,
        "plagiarism_score": plagiarism_score,
        "ai_score": ai_score,
        "verdict": verdict,
        "matched_sources": ",".join(matched_sources) if matched_sources else "",
        "upload_date": upload_date,
    }

    if fingerprint_similarity is not None:
        metadata["fingerprint_similarity"] = fingerprint_similarity

    if extra_metadata:
        metadata.update(extra_metadata)

    doc_id = _record_id(report_id)

    collection.add(
        ids=[doc_id],
        documents=[filename],   # store filename as the searchable document text
        metadatas=[metadata],
    )

    return doc_id


def list_analyses(limit: int = 100) -> list[dict[str, Any]]:
    """Return recent analysis records, newest first.

    Parameters
    ----------
    limit:
        Maximum number of records to return.

    Returns
    -------
    list of dict
        Each dict contains all metadata fields stored by :func:`add_analysis`.
    """
    collection = get_collection(COLLECTION_HISTORY)
    if collection.count() == 0:
        return []

    raw = collection.get(include=["metadatas"])
    records = list(raw["metadatas"])
    records.sort(key=lambda r: r.get("upload_date", ""), reverse=True)
    return records[:limit]


def get_analysis(report_id: str) -> dict[str, Any] | None:
    """Fetch a single analysis record by its report ID.

    Parameters
    ----------
    report_id:
        The report ID supplied when the record was created.

    Returns
    -------
    dict or None
        The metadata dict, or ``None`` if no matching record exists.
    """
    collection = get_collection(COLLECTION_HISTORY)
    doc_id = _record_id(report_id)

    try:
        raw = collection.get(ids=[doc_id], include=["metadatas"])
    except Exception:
        return None

    if not raw["metadatas"]:
        return None

    return raw["metadatas"][0]


def delete_analysis(report_id: str) -> bool:
    """Delete the analysis record for *report_id*.

    Returns
    -------
    bool
        ``True`` if the record existed and was deleted, ``False`` otherwise.
    """
    collection = get_collection(COLLECTION_HISTORY)
    doc_id = _record_id(report_id)

    try:
        raw = collection.get(ids=[doc_id], include=["metadatas"])
    except Exception:
        return False

    if not raw["metadatas"]:
        return False

    collection.delete(ids=[doc_id])
    return True


def analysis_count() -> int:
    """Return the total number of analysis records stored."""
    return get_collection(COLLECTION_HISTORY).count()
