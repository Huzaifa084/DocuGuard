"""
feature_extractor.py
--------------------
Stylometric and statistical feature extraction for DocuGuard+.

Produces a dictionary of measurable "fingerprint" values that feed into the
AI Detection Engine and the Writing Fingerprint Comparison module.

Feature groups:
  • Sentence metrics  – length mean / variance / burstiness
  • Vocabulary        – Type-Token Ratio (TTR), hapax ratio, lexical diversity
  • Grammar patterns  – passive voice ratio, stopword frequency
  • Style indicators  – punctuation patterns, repetition frequency
  • Readability       – Flesch Reading Ease
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List

import nltk
from nltk.corpus import stopwords

from utils.text_utils import (
    count_syllables,
    flesch_reading_ease,
    split_sentences,
    tokenize_words,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_STOPWORDS: set[str] = set(stopwords.words("english"))

_TRANSITION_WORDS: set[str] = {
    "however", "therefore", "moreover", "furthermore", "additionally",
    "consequently", "nevertheless", "meanwhile", "nonetheless",
    "subsequently", "accordingly", "hence", "thus", "indeed",
    "alternatively", "conversely", "likewise", "similarly",
    "specifically", "notably", "importantly", "significantly",
    "ultimately", "essentially", "overall", "in conclusion",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(text: str) -> Dict[str, Any]:
    """Return a flat dictionary of stylometric / statistical features.

    Parameters
    ----------
    text:
        Cleaned, full-document text.

    Returns
    -------
    dict
        Keys are human-readable feature names; values are floats / ints.
    """
    sentences = split_sentences(text)
    words = tokenize_words(text)

    features: Dict[str, Any] = {}

    # ---- Sentence metrics ------------------------------------------------
    features.update(_sentence_metrics(sentences, words))

    # ---- Vocabulary metrics ----------------------------------------------
    features.update(_vocabulary_metrics(words))

    # ---- Grammar patterns --------------------------------------------------
    features.update(_grammar_patterns(text, words, sentences))

    # ---- Style indicators --------------------------------------------------
    features.update(_style_indicators(text, sentences, words))

    # ---- Paragraph metrics (used by expanded StyleAnalyser) ---------------
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if paragraphs:
        para_lengths = [len(p.split()) for p in paragraphs]
        para_mean = sum(para_lengths) / len(para_lengths)
        para_std = math.sqrt(
            sum((l - para_mean) ** 2 for l in para_lengths) / len(para_lengths)
        ) if len(para_lengths) > 1 else 0.0
        features["paragraph_length_mean"] = round(para_mean, 3)
        features["paragraph_length_std"] = round(para_std, 3)
    else:
        features["paragraph_length_mean"] = 0.0
        features["paragraph_length_std"] = 0.0

    # ---- Readability -------------------------------------------------------
    features["flesch_reading_ease"] = flesch_reading_ease(text)

    return features


# Features excluded from the ML vector — document-length artifacts, not style
_EXCLUDE_FROM_VECTOR: set[str] = {"sentence_count"}


def extract_feature_vector(text: str) -> List[float]:
    """Return ordered, z-score-normalised numeric feature values for ML.

    Excludes document-length artefacts (``sentence_count``) and applies
    z-score normalisation so features with vastly different scales
    (e.g. Yule's K 0-200 vs ratios 0-1) don't dominate the classifier.
    """
    feats = extract_features(text)
    # Sort by key for deterministic ordering, exclude non-style features
    raw = [
        float(v)
        for k, v in sorted(feats.items())
        if isinstance(v, (int, float)) and k not in _EXCLUDE_FROM_VECTOR
    ]
    # Z-score normalise (robust to zero-variance features)
    if raw:
        import numpy as _np
        arr = _np.array(raw, dtype=_np.float64)
        mu = arr.mean()
        sigma = arr.std()
        if sigma > 1e-9:
            arr = (arr - mu) / sigma
        raw = arr.tolist()
    return raw


def feature_names() -> List[str]:
    """Return the sorted list of numeric feature names (matches vector order)."""
    # Build a dummy feature dict to discover keys
    dummy = extract_features("The quick brown fox jumps over the lazy dog. " * 5)
    return sorted(
        k for k, v in dummy.items()
        if isinstance(v, (int, float)) and k not in _EXCLUDE_FROM_VECTOR
    )


# ---------------------------------------------------------------------------
# Sentence metrics
# ---------------------------------------------------------------------------

def _sentence_metrics(sentences: List[str], words: List[str]) -> Dict[str, float]:
    if not sentences:
        return {
            "sentence_count": 0,
            "sentence_length_mean": 0.0,
            "sentence_length_variance": 0.0,
            "sentence_length_std": 0.0,
            "burstiness": 0.0,
        }

    lengths = [len(s.split()) for s in sentences]
    n = len(lengths)
    mean = sum(lengths) / n
    variance = sum((l - mean) ** 2 for l in lengths) / n if n > 1 else 0.0
    std = math.sqrt(variance)

    # Burstiness: coefficient of variation (std / mean). High → human-like.
    burstiness = std / mean if mean > 0 else 0.0

    return {
        "sentence_count": n,
        "sentence_length_mean": round(mean, 3),
        "sentence_length_variance": round(variance, 3),
        "sentence_length_std": round(std, 3),
        "burstiness": round(burstiness, 3),
    }


# ---------------------------------------------------------------------------
# Vocabulary metrics
# ---------------------------------------------------------------------------

def _vocabulary_metrics(words: List[str]) -> Dict[str, float]:
    if not words:
        return {
            "type_token_ratio": 0.0,
            "hapax_ratio": 0.0,
            "lexical_diversity": 0.0,
            "avg_word_length": 0.0,
            "long_word_ratio": 0.0,
        }

    freq = Counter(words)
    types = len(freq)
    tokens = len(words)
    hapax = sum(1 for c in freq.values() if c == 1)

    ttr = types / tokens
    hapax_ratio = hapax / tokens
    # Yule's K approximation for lexical diversity
    lexical_diversity = _yules_k(freq, tokens)
    avg_word_length = sum(len(w) for w in words) / tokens
    long_word_ratio = sum(1 for w in words if len(w) > 6) / tokens

    return {
        "type_token_ratio": round(ttr, 4),
        "hapax_ratio": round(hapax_ratio, 4),
        "lexical_diversity": round(lexical_diversity, 4),
        "avg_word_length": round(avg_word_length, 3),
        "long_word_ratio": round(long_word_ratio, 4),
    }


def _yules_k(freq: Counter, n: int) -> float:
    """Yule's K — a measure of vocabulary richness.

    Lower values indicate *more* diverse vocabulary.
    """
    if n == 0:
        return 0.0
    spectrum = Counter(freq.values())
    m2 = sum(i * i * vi for i, vi in spectrum.items())
    k = 10000 * (m2 - n) / (n * n) if n > 1 else 0.0
    return k


# ---------------------------------------------------------------------------
# Grammar patterns
# ---------------------------------------------------------------------------

def _grammar_patterns(text: str, words: List[str], sentences: List[str]) -> Dict[str, float]:
    if not words:
        return {
            "passive_voice_ratio": 0.0,
            "stopword_ratio": 0.0,
            "modal_verb_ratio": 0.0,
            "conjunction_ratio": 0.0,
        }

    # ---- Passive voice detection (batch POS-tag) ---
    passive_count = _count_passive_voice_batch(sentences)
    passive_ratio = passive_count / len(sentences) if sentences else 0.0

    # ---- Stopword frequency --
    sw_count = sum(1 for w in words if w in _STOPWORDS)
    stopword_ratio = sw_count / len(words)

    # ---- Modal verbs --
    modals = {"can", "could", "may", "might", "must", "shall", "should",
              "will", "would"}
    modal_count = sum(1 for w in words if w in modals)
    modal_ratio = modal_count / len(words)

    # ---- Coordinating conjunctions --
    conjunctions = {"and", "but", "or", "nor", "for", "yet", "so"}
    conj_count = sum(1 for w in words if w in conjunctions)
    conj_ratio = conj_count / len(words)

    return {
        "passive_voice_ratio": round(passive_ratio, 4),
        "stopword_ratio": round(stopword_ratio, 4),
        "modal_verb_ratio": round(modal_ratio, 4),
        "conjunction_ratio": round(conj_ratio, 4),
    }


def _count_passive_voice_batch(sentences: List[str]) -> int:
    """Batch-POS-tag all sentences at once, then detect passive constructions.

    Previous implementation called ``nltk.pos_tag()`` per sentence (N calls).
    This batches everything into a single tag operation, then splits by
    sentence boundaries.
    """
    if not sentences:
        return 0

    be_forms = {"am", "is", "are", "was", "were", "be", "been", "being"}
    passive_count = 0

    # Tokenise all sentences first, remember boundaries
    all_tokens: List[str] = []
    boundaries: List[int] = []  # end index of each sentence's tokens
    for sent in sentences:
        try:
            tokens = nltk.word_tokenize(sent)
        except Exception:
            tokens = sent.split()
        all_tokens.extend(tokens)
        boundaries.append(len(all_tokens))

    if not all_tokens:
        return 0

    # Single POS-tag call for the whole document
    try:
        all_tagged = nltk.pos_tag(all_tokens)
    except Exception:
        return 0

    # Walk through sentence boundaries and detect passive voice
    start = 0
    for end in boundaries:
        tagged_sent = all_tagged[start:end]
        for i in range(len(tagged_sent) - 1):
            word, _tag = tagged_sent[i]
            _next_word, next_tag = tagged_sent[i + 1]
            if word.lower() in be_forms and next_tag == "VBN":
                passive_count += 1
                break  # count once per sentence
        start = end

    return passive_count


# ---------------------------------------------------------------------------
# Style indicators
# ---------------------------------------------------------------------------

def _style_indicators(
    text: str, sentences: List[str], words: List[str]
) -> Dict[str, float]:
    if not words:
        return {
            "comma_ratio": 0.0,
            "semicolon_ratio": 0.0,
            "question_ratio": 0.0,
            "exclamation_ratio": 0.0,
            "transition_word_ratio": 0.0,
            "repeated_phrase_ratio": 0.0,
            "contraction_ratio": 0.0,
            "sentence_starter_diversity": 0.0,
        }

    n_words = len(words)
    n_sents = max(len(sentences), 1)

    # ---- Punctuation ratios (per word) --
    comma_ratio = text.count(",") / n_words
    semicolon_ratio = text.count(";") / n_words
    question_ratio = text.count("?") / n_sents
    exclamation_ratio = text.count("!") / n_sents

    # ---- Transition word frequency --
    transition_count = sum(1 for w in words if w in _TRANSITION_WORDS)
    transition_ratio = transition_count / n_words

    # ---- Repeated n-gram ratio (bigrams appearing > 2 times) --
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    bigram_freq = Counter(bigrams)
    repeated = sum(c for c in bigram_freq.values() if c > 2)
    repeated_phrase_ratio = repeated / max(len(bigrams), 1)

    # ---- Contractions (hint of informal / human writing) --
    contraction_count = len(re.findall(
        r"\b\w+n't\b|\b\w+'(?:re|ve|ll|s|d|m)\b", text, re.IGNORECASE
    ))
    contraction_ratio = contraction_count / n_words

    # ---- Sentence starter diversity --
    starters = [s.split()[0].lower() for s in sentences if s.split()]
    unique_starters = len(set(starters))
    starter_diversity = unique_starters / len(starters) if starters else 0.0

    return {
        "comma_ratio": round(comma_ratio, 4),
        "semicolon_ratio": round(semicolon_ratio, 4),
        "question_ratio": round(question_ratio, 4),
        "exclamation_ratio": round(exclamation_ratio, 4),
        "transition_word_ratio": round(transition_ratio, 4),
        "repeated_phrase_ratio": round(repeated_phrase_ratio, 4),
        "contraction_ratio": round(contraction_ratio, 4),
        "sentence_starter_diversity": round(starter_diversity, 4),
    }
