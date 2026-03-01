"""
plagiarism_detector.py
----------------------
Plagiarism Analysis Engine for DocuGuard+.

Two complementary detection strategies:

1. **Lexical Similarity** – TF-IDF vectorisation + cosine similarity to find
   direct textual overlap.
2. **Semantic Similarity** – Sentence-Transformer embeddings to detect
   paraphrased content where the meaning is preserved but wording differs.

Both compare the input text against the local document corpus stored in
ChromaDB.  A clear disclaimer is included: detection is limited to the
local reference corpus.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from db.corpus_store import get_all_texts, query_similar
from utils.text_utils import split_sentences

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MatchSegment:
    """A single flagged text segment and its best corpus match."""
    input_text: str = ""
    matched_text: str = ""
    source_filename: str = ""
    similarity: float = 0.0
    match_type: str = ""  # "lexical" or "semantic"


@dataclass
class PlagiarismResult:
    """Aggregated plagiarism analysis output."""
    overall_score: float = 0.0            # 0 = clean, 1 = fully matched
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    matched_segments: List[MatchSegment] = field(default_factory=list)
    matched_sources: List[str] = field(default_factory=list)
    disclaimer: str = (
        "Plagiarism detection is limited to the local reference corpus. "
        "A low score does NOT guarantee originality against all published work."
    )


# ---------------------------------------------------------------------------
# Sentence-Transformer loader (lazy)
# ---------------------------------------------------------------------------

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        cache = os.environ.get("TRANSFORMERS_CACHE", None)
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME, cache_folder=cache)
    return _embed_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PlagiarismDetector:
    """Compare an input document against the local corpus."""

    def __init__(
        self,
        lexical_threshold: float = 0.40,
        semantic_threshold: float = 0.75,
    ) -> None:
        self.lexical_threshold = lexical_threshold
        self.semantic_threshold = semantic_threshold

    def check(self, text: str) -> PlagiarismResult:
        """Run both lexical and semantic plagiarism checks.

        Parameters
        ----------
        text :
            Cleaned full-document text to analyse.

        Returns
        -------
        PlagiarismResult

        Notes
        -----
        **Semantic check** now delegates to ChromaDB's native vector search
        (``query_similar``) instead of re-encoding the entire corpus on every
        call.  This reduces runtime from O(N×M) to O(N log M).

        **Scoring** uses a *coverage-based* scheme: the score reflects the
        **fraction of input sentences that exceed the threshold**, not the
        average similarity across all sentences (which dilutes real matches).
        """
        corpus_pairs = get_all_texts()  # list of (chunk_text, filename)
        if not corpus_pairs:
            return PlagiarismResult(
                disclaimer=(
                    "The local corpus is empty.  Upload reference documents "
                    "to enable plagiarism comparison."
                )
            )

        corpus_texts = [t for t, _ in corpus_pairs]
        corpus_files = [f for _, f in corpus_pairs]

        input_sentences = split_sentences(text)
        if not input_sentences:
            return PlagiarismResult()

        # 1. Lexical analysis (still uses TF-IDF — fast & complementary)
        lex_matches, lex_score = self._lexical_check(
            input_sentences, corpus_texts, corpus_files
        )

        # 2. Semantic analysis — ChromaDB-native vector search
        sem_matches, sem_score = self._semantic_check_chromadb(
            input_sentences
        )

        # Merge & deduplicate
        all_matches = lex_matches + sem_matches
        sources = sorted({m.source_filename for m in all_matches})

        # Combined score (max of both, weighted slightly towards semantic)
        overall = 0.4 * lex_score + 0.6 * sem_score

        return PlagiarismResult(
            overall_score=round(overall, 4),
            lexical_score=round(lex_score, 4),
            semantic_score=round(sem_score, 4),
            matched_segments=all_matches,
            matched_sources=sources,
        )

    # ------------------------------------------------------------------
    # Lexical: TF-IDF cosine
    # ------------------------------------------------------------------

    def _lexical_check(
        self,
        input_sentences: List[str],
        corpus_texts: List[str],
        corpus_files: List[str],
    ) -> Tuple[List[MatchSegment], float]:
        """TF-IDF cosine similarity check with **coverage-based** scoring."""
        all_docs = corpus_texts + input_sentences
        vectorizer = TfidfVectorizer(
            max_features=20_000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(all_docs)
        except ValueError:
            return [], 0.0

        n_corpus = len(corpus_texts)
        corpus_vecs = tfidf_matrix[:n_corpus]
        input_vecs = tfidf_matrix[n_corpus:]

        sim_matrix = cosine_similarity(input_vecs, corpus_vecs)

        matches: List[MatchSegment] = []
        flagged_count = 0

        for i, sent in enumerate(input_sentences):
            best_j = int(np.argmax(sim_matrix[i]))
            best_sim = float(sim_matrix[i, best_j])

            if best_sim >= self.lexical_threshold:
                flagged_count += 1
                matches.append(MatchSegment(
                    input_text=sent,
                    matched_text=corpus_texts[best_j][:300],
                    source_filename=corpus_files[best_j],
                    similarity=round(best_sim, 4),
                    match_type="lexical",
                ))

        # Coverage-based: fraction of sentences that exceeded the threshold
        coverage_score = flagged_count / len(input_sentences) if input_sentences else 0.0
        return matches, coverage_score

    # ------------------------------------------------------------------
    # Semantic: ChromaDB-native vector search (replaces brute-force)
    # ------------------------------------------------------------------

    def _semantic_check_chromadb(
        self,
        input_sentences: List[str],
    ) -> Tuple[List[MatchSegment], float]:
        """ChromaDB-native semantic similarity check.

        Uses ``query_similar()`` per sentence, leveraging the pre-computed
        embeddings stored in ChromaDB.  This is O(N log M) versus re-encoding
        the full corpus on every call.
        """
        matches: List[MatchSegment] = []
        flagged_count = 0

        for sent in input_sentences:
            # Skip very short sentences (< 5 words) — too noisy
            if len(sent.split()) < 5:
                continue

            results = query_similar(sent, n_results=1)
            if not results:
                continue

            top = results[0]
            # ChromaDB returns L2 distance — convert to cosine similarity.
            # For unit-normalised vectors: cos_sim = 1 - dist^2 / 2
            # The default embedding function normalises, so this is safe.
            distance = top.get("distance", 2.0)
            similarity = max(0.0, 1.0 - distance / 2.0)

            if similarity >= self.semantic_threshold:
                flagged_count += 1
                matches.append(MatchSegment(
                    input_text=sent,
                    matched_text=top.get("document", "")[:300],
                    source_filename=top.get("filename", "unknown"),
                    similarity=round(similarity, 4),
                    match_type="semantic",
                ))

        n_checked = max(
            1, sum(1 for s in input_sentences if len(s.split()) >= 5)
        )
        coverage_score = flagged_count / n_checked
        return matches, coverage_score

    # ------------------------------------------------------------------
    # Legacy semantic: kept for offline/batch use if needed
    # ------------------------------------------------------------------

    def _semantic_check(
        self,
        input_sentences: List[str],
        corpus_texts: List[str],
        corpus_files: List[str],
    ) -> Tuple[List[MatchSegment], float]:
        """Sentence-embedding cosine similarity check."""
        model = _get_embed_model()

        # Encode in batches
        input_embs = model.encode(input_sentences, show_progress_bar=False)
        corpus_embs = model.encode(corpus_texts, show_progress_bar=False,
                                   batch_size=64)

        sim_matrix = cosine_similarity(input_embs, corpus_embs)

        matches: List[MatchSegment] = []
        scores: List[float] = []

        for i, sent in enumerate(input_sentences):
            best_j = int(np.argmax(sim_matrix[i]))
            best_sim = float(sim_matrix[i, best_j])
            scores.append(best_sim)

            if best_sim >= self.semantic_threshold:
                matches.append(MatchSegment(
                    input_text=sent,
                    matched_text=corpus_texts[best_j][:300],
                    source_filename=corpus_files[best_j],
                    similarity=round(best_sim, 4),
                    match_type="semantic",
                ))

        avg_score = float(np.mean(scores)) if scores else 0.0
        return matches, avg_score


# ---------------------------------------------------------------------------
# ChromaDB-backed quick semantic search (convenience wrapper)
# ---------------------------------------------------------------------------

def quick_semantic_search(
    text: str, n_results: int = 5
) -> List[Dict[str, Any]]:
    """Find the most similar corpus chunks to *text* via ChromaDB embeddings.

    This uses ChromaDB's built-in embedding function (default: all-MiniLM-L6-v2).
    """
    return query_similar(text, n_results=n_results)
