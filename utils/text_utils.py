"""
text_utils.py
-------------
Common text-processing helpers used across the DocuGuard+ pipeline.
"""

import re
import unicodedata
from typing import List

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# ---------------------------------------------------------------------------
# Ensure NLTK data is available (silent no-op when already present)
# ---------------------------------------------------------------------------
for _pkg in ("punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger",
             "averaged_perceptron_tagger_eng", "wordnet"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}" if "punkt" in _pkg else _pkg)
    except LookupError:
        nltk.download(_pkg, quiet=True)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def clean_text(raw: str) -> str:
    """Normalise Unicode, collapse whitespace, strip non-printable chars."""
    text = unicodedata.normalize("NFKC", raw)
    # Remove non-printable / control characters except newline and tab
    text = re.sub(r"[^\S\n\t]+", " ", text)
    # Collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """Split *text* into sentences using NLTK's Punkt tokeniser."""
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]


def split_paragraphs(text: str) -> List[str]:
    """Split on double-newlines (or more) to extract paragraphs."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def tokenize_words(text: str) -> List[str]:
    """Return lowercased word tokens (no punctuation-only tokens)."""
    tokens = word_tokenize(text)
    return [t.lower() for t in tokens if re.search(r"[a-zA-Z0-9]", t)]


def count_words(text: str) -> int:
    """Fast whitespace-based word count."""
    return len(text.split())


def count_syllables(word: str) -> int:
    """Estimate syllable count using a vowel-group heuristic."""
    word = word.lower().strip()
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Adjust for silent 'e'
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def flesch_reading_ease(text: str) -> float:
    """Compute the Flesch Reading Ease score for *text*.

    Higher → easier to read (typical AI text scores 50-70).
    """
    sentences = split_sentences(text)
    words = tokenize_words(text)
    if not sentences or not words:
        return 0.0
    total_syllables = sum(count_syllables(w) for w in words)
    asl = len(words) / len(sentences)
    asw = total_syllables / len(words)
    return 206.835 - 1.015 * asl - 84.6 * asw
