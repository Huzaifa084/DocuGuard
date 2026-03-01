"""
report_generator.py
-------------------
Explainable Report Generator for DocuGuard+.

Synthesises outputs from every analysis engine into a single structured report
containing:

  • Document Metadata (word counts, sentence counts, etc.)
  • AI Detection Insights (probability, explanations, contributing factors)
  • Plagiarism Map (highlighted segments, lexical vs semantic matches)
  • Humanization Metrics (before / after AI score comparison)
  • Fingerprint Comparison (style similarity when a profile exists)

Reports are returned as dictionaries and can optionally be persisted to a
JSON file under ``reports/``.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.ai_detector import AIDetectionResult
from core.document_processor import DocumentMetadata
from core.fingerprint import FingerprintResult
from core.humanizer import HumanizationResult
from core.plagiarism_detector import PlagiarismResult

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPORT_DIR = os.environ.get("REPORT_DIR", "reports")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_report(
    metadata: DocumentMetadata,
    ai_result: AIDetectionResult,
    plagiarism_result: Optional[PlagiarismResult] = None,
    humanization_result: Optional[HumanizationResult] = None,
    fingerprint_result: Optional[FingerprintResult] = None,
    report_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the final explainable report dictionary.

    Parameters
    ----------
    metadata :
        Basic document statistics.
    ai_result :
        Output from :class:`core.ai_detector.AIDetector`.
    plagiarism_result :
        Output from :class:`core.plagiarism_detector.PlagiarismDetector`.
    humanization_result :
        Output from :class:`core.humanizer.Humanizer` (if humanization ran).
    fingerprint_result :
        Output from :class:`core.fingerprint.FingerprintEngine` (if profile
        comparison ran).
    report_id :
        Caller-supplied unique ID.  Auto-generated if ``None``.

    Returns
    -------
    dict
        Full report as a JSON-serialisable dictionary.
    """
    rid = report_id or uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()

    report: Dict[str, Any] = {
        "report_id": rid,
        "generated_at": now,
        "disclaimer": (
            "AI detection is probabilistic, not definitive. This tool is "
            "intended for academic experimentation and writing improvement. "
            "Treat results as indicators, not absolute proof of origin."
        ),
    }

    # ---- Document Metadata ------------------------------------------------
    report["document_metadata"] = {
        "filename": metadata.filename,
        "word_count": metadata.word_count,
        "sentence_count": metadata.sentence_count,
        "paragraph_count": metadata.paragraph_count,
        "character_count": metadata.char_count,
        "avg_sentence_length": round(metadata.avg_sentence_length, 2),
    }

    # ---- AI Detection Insights --------------------------------------------
    report["ai_detection"] = {
        "ai_probability": ai_result.ai_probability,
        "confidence": ai_result.confidence,
        "perplexity": {
            "value": ai_result.perplexity_result.perplexity,
            "normalised_score": ai_result.perplexity_result.normalised_score,
            "interpretation": ai_result.perplexity_result.interpretation,
        },
        "style_analysis": {
            "uniformity_score": ai_result.style_result.uniformity_score,
            "vocabulary_score": ai_result.style_result.vocabulary_score,
            "naturalness_score": ai_result.style_result.naturalness_score,
            "contributing_factors": ai_result.style_result.contributing_factors,
        },
        "explanations": ai_result.explanations,
        "key_features": _select_key_features(ai_result.features),
    }

    # ---- Plagiarism Map ---------------------------------------------------
    if plagiarism_result is not None:
        segments = []
        for seg in plagiarism_result.matched_segments[:20]:  # cap displayed
            segments.append({
                "input_excerpt": seg.input_text[:200],
                "matched_excerpt": seg.matched_text[:200],
                "source": seg.source_filename,
                "similarity": seg.similarity,
                "match_type": seg.match_type,
            })

        report["plagiarism"] = {
            "overall_score": plagiarism_result.overall_score,
            "lexical_score": plagiarism_result.lexical_score,
            "semantic_score": plagiarism_result.semantic_score,
            "matched_sources": plagiarism_result.matched_sources,
            "matched_segments": segments,
            "disclaimer": plagiarism_result.disclaimer,
        }

    # ---- Humanization Metrics (Before vs After) ---------------------------
    if humanization_result is not None:
        report["humanization"] = {
            "strategy": humanization_result.strategy,
            "changes_summary": humanization_result.changes_summary,
            "ai_probability_before": humanization_result.ai_prob_before,
            "ai_probability_after": humanization_result.ai_prob_after,
            "improvement": (
                round(
                    (humanization_result.ai_prob_before or 0)
                    - (humanization_result.ai_prob_after or 0),
                    4,
                )
                if humanization_result.ai_prob_before is not None
                   and humanization_result.ai_prob_after is not None
                else None
            ),
        }

    # ---- Fingerprint Comparison -------------------------------------------
    if fingerprint_result is not None:
        report["fingerprint"] = {
            "profile_name": fingerprint_result.profile_name,
            "similarity_score": fingerprint_result.similarity_score,
            "interpretation": fingerprint_result.interpretation,
            "top_deviations": dict(
                sorted(
                    fingerprint_result.deviation_summary.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[:5]
            ),
        }

    return report


def save_report(report: Dict[str, Any]) -> str:
    """Persist *report* as a JSON file in ``REPORT_DIR``.

    Returns
    -------
    str
        Absolute path to the saved file.
    """
    os.makedirs(REPORT_DIR, exist_ok=True)
    rid = report.get("report_id", uuid.uuid4().hex)
    path = os.path.join(REPORT_DIR, f"report_{rid}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return os.path.abspath(path)


def load_report(report_id: str) -> Optional[Dict[str, Any]]:
    """Load a previously saved report by its ID."""
    path = os.path.join(REPORT_DIR, f"report_{report_id}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_reports() -> List[Dict[str, str]]:
    """Return metadata for all saved reports, newest first."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    reports: List[Dict[str, str]] = []
    for fname in os.listdir(REPORT_DIR):
        if fname.startswith("report_") and fname.endswith(".json"):
            rid = fname.removeprefix("report_").removesuffix(".json")
            path = os.path.join(REPORT_DIR, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                reports.append({
                    "report_id": rid,
                    "filename": data.get("document_metadata", {}).get("filename", ""),
                    "generated_at": data.get("generated_at", ""),
                    "ai_probability": str(
                        data.get("ai_detection", {}).get("ai_probability", "")
                    ),
                })
            except Exception:
                continue
    reports.sort(key=lambda r: r.get("generated_at", ""), reverse=True)
    return reports


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KEY_FEATURES = [
    "burstiness",
    "type_token_ratio",
    "sentence_length_mean",
    "sentence_length_variance",
    "passive_voice_ratio",
    "stopword_ratio",
    "transition_word_ratio",
    "contraction_ratio",
    "sentence_starter_diversity",
    "flesch_reading_ease",
]


def _select_key_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the most explanatory features for the report."""
    return {k: features[k] for k in _KEY_FEATURES if k in features}
