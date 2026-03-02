"""
history_store.py
----------------
Persistence layer for document analysis results — **SQLite backend**.

Previous versions used ChromaDB for this tabular data, but analysis history
is relational (report ID, filename, scores, dates) and ChromaDB cannot sort,
paginate, or aggregate.  SQLite is built-in, zero-dependency, and ideal for
this workload.

The database file lives at ``data/history.db`` by default (overridable via
the ``HISTORY_DB_PATH`` environment variable).
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DEFAULT_DB_DIR = os.path.join("data")
_DB_PATH = os.environ.get(
    "HISTORY_DB_PATH",
    os.path.join(_DEFAULT_DB_DIR, "history.db"),
)


def _get_connection() -> sqlite3.Connection:
    """Return a connection to the history database, creating it if needed."""
    os.makedirs(os.path.dirname(_DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the analyses table if it doesn't already exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id       TEXT    NOT NULL UNIQUE,
            filename        TEXT    NOT NULL,
            plagiarism_score REAL   NOT NULL DEFAULT 0.0,
            ai_score        REAL   NOT NULL DEFAULT 0.0,
            verdict         TEXT   NOT NULL DEFAULT '',
            matched_sources TEXT   NOT NULL DEFAULT '',
            fingerprint_similarity REAL,
            upload_date     TEXT   NOT NULL,
            extra_metadata  TEXT   NOT NULL DEFAULT '{}'
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_analyses_upload_date
        ON analyses(upload_date DESC)
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Public API — same signatures as the old ChromaDB version
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

    Returns the report_id (used as the logical key).
    """
    upload_date = datetime.now(timezone.utc).isoformat()
    sources_csv = ",".join(matched_sources) if matched_sources else ""
    extra_json = json.dumps(extra_metadata or {})

    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO analyses
                (report_id, filename, plagiarism_score, ai_score, verdict,
                 matched_sources, fingerprint_similarity, upload_date,
                 extra_metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report_id,
                filename,
                plagiarism_score,
                ai_score,
                verdict,
                sources_csv,
                fingerprint_similarity,
                upload_date,
                extra_json,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return report_id


def list_analyses(limit: int = 100) -> list[dict[str, Any]]:
    """Return recent analysis records, newest first."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM analyses ORDER BY upload_date DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    return [_row_to_dict(r) for r in rows]


def get_analysis(report_id: str) -> dict[str, Any] | None:
    """Fetch a single analysis record by its report ID."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM analyses WHERE report_id = ?", (report_id,)
        ).fetchone()
    finally:
        conn.close()

    return _row_to_dict(row) if row else None


def delete_analysis(report_id: str) -> bool:
    """Delete the analysis record for *report_id*."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "DELETE FROM analyses WHERE report_id = ?", (report_id,)
        )
        conn.commit()
        deleted = cursor.rowcount > 0
    finally:
        conn.close()
    return deleted


def analysis_count() -> int:
    """Return the total number of analysis records stored."""
    conn = _get_connection()
    try:
        (count,) = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()
    finally:
        conn.close()
    return count


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a Row to the same dict shape the old ChromaDB version returned."""
    d = dict(row)
    # Parse matched_sources back to list
    sources_str = d.pop("matched_sources", "")
    d["matched_sources"] = sources_str.split(",") if sources_str else []
    # Parse extra metadata
    extra_str = d.pop("extra_metadata", "{}")
    try:
        extra = json.loads(extra_str)
    except (json.JSONDecodeError, TypeError):
        extra = {}
    d.update(extra)
    # Remove the auto-increment id — callers don't use it
    d.pop("id", None)
    return d
