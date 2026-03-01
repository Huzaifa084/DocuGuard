"""
humanizer.py
------------
Humanization Engine (Controlled Naturalization) for DocuGuard+.

Provides two strategies:

1. **Rule-Based** — deterministic transformations (sentence splitting,
   contraction injection, transition variation, length perturbation).
   Works fully offline with zero latency.

2. **LLM-Based** — sends text to a local Ollama instance for intelligent
   rewriting that preserves meaning while improving natural flow.

After humanization the AI Detection Engine is re-run automatically to produce
a "Before vs. After" probability comparison.
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass, field
from typing import List, Optional

import requests

from utils.text_utils import split_sentences

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("DEFAULT_OLLAMA_MODEL", "mistral")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HumanizationResult:
    """Output of the humanization engine."""
    original_text: str = ""
    humanized_text: str = ""
    strategy: str = ""                  # "rule-based" | "llm"
    changes_summary: List[str] = field(default_factory=list)
    ai_prob_before: Optional[float] = None
    ai_prob_after: Optional[float] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Humanizer:
    """Controlled text naturalization with optional LLM rewriting."""

    def humanize(
        self,
        text: str,
        strategy: str = "rule-based",
        model: str = DEFAULT_MODEL,
    ) -> HumanizationResult:
        """Rewrite *text* to improve naturalness.

        Parameters
        ----------
        text :
            The cleaned document text.
        strategy :
            ``"rule-based"`` (default, offline) or ``"llm"`` (requires Ollama).
        model :
            Ollama model name when *strategy* is ``"llm"``.

        Returns
        -------
        HumanizationResult
            Contains original, rewritten text, and a summary of changes.
        """
        if strategy == "llm":
            return self._llm_humanize(text, model)
        return self._rule_based_humanize(text)

    # ------------------------------------------------------------------
    # Rule-based transforms
    # ------------------------------------------------------------------

    def _rule_based_humanize(self, text: str) -> HumanizationResult:
        changes: List[str] = []
        sentences = split_sentences(text)
        if not sentences:
            return HumanizationResult(
                original_text=text, humanized_text=text,
                strategy="rule-based", changes_summary=["No changes (empty text)."]
            )

        new_sentences: List[str] = []

        for sent in sentences:
            s = sent

            # 1. Replace overused transitions with varied alternatives
            s, t_changed = _vary_transitions(s)
            if t_changed:
                changes.append("Varied transition words")

            # 2. Inject contractions where natural
            s, c_changed = _inject_contractions(s)
            if c_changed:
                changes.append("Injected contractions")

            # 3. Occasionally split long sentences
            if len(s.split()) > 25 and random.random() < 0.4:
                parts = _split_long_sentence(s)
                if len(parts) > 1:
                    new_sentences.extend(parts)
                    changes.append("Split a long sentence")
                    continue

            # 4. Occasionally merge short consecutive sentences
            if (
                new_sentences
                and len(new_sentences[-1].split()) < 8
                and len(s.split()) < 8
                and random.random() < 0.3
            ):
                merged = new_sentences.pop().rstrip(".")
                s = f"{merged}, and {s[0].lower()}{s[1:]}"
                changes.append("Merged two short sentences")

            new_sentences.append(s)

        # 5. Minor length perturbation (add filler or trim)
        new_sentences = _perturb_lengths(new_sentences)
        if len(new_sentences) != len(sentences):
            changes.append("Adjusted sentence lengths for variability")

        humanized = " ".join(new_sentences)

        # Deduplicate change log
        changes = list(dict.fromkeys(changes))

        return HumanizationResult(
            original_text=text,
            humanized_text=humanized,
            strategy="rule-based",
            changes_summary=changes if changes else ["Minor phrasing adjustments applied."],
        )

    # ------------------------------------------------------------------
    # LLM-based rewriting (Ollama)
    # ------------------------------------------------------------------

    def _llm_humanize(self, text: str, model: str) -> HumanizationResult:
        prompt = (
            "You are a professional academic editor. Rewrite the following text "
            "to sound more naturally human-written while preserving ALL factual "
            "content and meaning. Guidelines:\n"
            "- Vary sentence lengths and structures\n"
            "- Use occasional contractions where natural\n"
            "- Reduce repetitive transition words\n"
            "- Maintain an academic but approachable tone\n"
            "- Do NOT add new information or opinions\n"
            "- Return ONLY the rewritten text, no commentary\n\n"
            f"TEXT:\n{text}"
        )

        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            rewritten = data.get("response", "").strip()
            if not rewritten:
                raise ValueError("Empty response from Ollama")
        except Exception as exc:
            # Fallback to rule-based on any failure
            result = self._rule_based_humanize(text)
            result.changes_summary.insert(
                0, f"LLM unavailable ({exc}); fell back to rule-based approach."
            )
            result.strategy = "rule-based (fallback)"
            return result

        return HumanizationResult(
            original_text=text,
            humanized_text=rewritten,
            strategy="llm",
            changes_summary=[f"Rewritten by {model} via Ollama."],
        )


# ---------------------------------------------------------------------------
# Rule-based helper functions
# ---------------------------------------------------------------------------

_TRANSITION_MAP: dict[str, list[str]] = {
    "however": ["yet", "still", "that said", "on the other hand"],
    "therefore": ["so", "as a result", "consequently", "thus"],
    "moreover": ["also", "in addition", "besides", "what's more"],
    "furthermore": ["also", "on top of that", "in addition"],
    "additionally": ["also", "plus", "beyond that"],
    "consequently": ["as a result", "so", "because of this"],
    "nevertheless": ["even so", "still", "yet", "regardless"],
    "subsequently": ["then", "after that", "later"],
    "specifically": ["in particular", "namely", "to be specific"],
    "significantly": ["notably", "remarkably", "meaningfully"],
    "ultimately": ["in the end", "eventually", "finally"],
    "essentially": ["basically", "at its core", "in essence"],
}

_CONTRACTION_MAP: dict[str, str] = {
    "do not": "don't", "does not": "doesn't", "did not": "didn't",
    "is not": "isn't", "are not": "aren't", "was not": "wasn't",
    "were not": "weren't", "will not": "won't", "would not": "wouldn't",
    "could not": "couldn't", "should not": "shouldn't",
    "it is": "it's", "that is": "that's",
    "they are": "they're", "we are": "we're",
    "I am": "I'm", "you are": "you're",
    "I have": "I've", "you have": "you've",
    "we have": "we've", "they have": "they've",
    "I will": "I'll", "you will": "you'll",
    "we will": "we'll", "they will": "they'll",
    "I would": "I'd", "you would": "you'd",
    "we would": "we'd", "they would": "they'd",
}


def _vary_transitions(sentence: str) -> tuple[str, bool]:
    changed = False
    lower = sentence.lower()
    for original, alternatives in _TRANSITION_MAP.items():
        if original in lower and random.random() < 0.6:
            replacement = random.choice(alternatives)
            # Preserve case of original
            pattern = re.compile(re.escape(original), re.IGNORECASE)
            match = pattern.search(sentence)
            if match:
                orig_text = match.group()
                if orig_text[0].isupper():
                    replacement = replacement[0].upper() + replacement[1:]
                sentence = pattern.sub(replacement, sentence, count=1)
                changed = True
    return sentence, changed


def _inject_contractions(sentence: str) -> tuple[str, bool]:
    changed = False
    for full, contracted in _CONTRACTION_MAP.items():
        if full.lower() in sentence.lower() and random.random() < 0.5:
            pattern = re.compile(re.escape(full), re.IGNORECASE)
            sentence = pattern.sub(contracted, sentence, count=1)
            changed = True
    return sentence, changed


def _split_long_sentence(sentence: str) -> List[str]:
    """Try to split a sentence at a conjunction or semicolon."""
    # Try splitting at ", and " or ", but " or "; "
    for delim in [", and ", ", but ", "; ", " — "]:
        if delim in sentence:
            parts = sentence.split(delim, 1)
            if len(parts) == 2 and len(parts[0].split()) > 5:
                a = parts[0].rstrip(",;").strip()
                b = parts[1].strip()
                if not a.endswith("."):
                    a += "."
                b = b[0].upper() + b[1:] if b else b
                return [a, b]
    return [sentence]


def _perturb_lengths(sentences: List[str]) -> List[str]:
    """Small random perturbations to sentence lengths for variability."""
    result: List[str] = []
    for sent in sentences:
        words = sent.split()
        # Occasionally add a mild interjection for short sentences
        if len(words) < 6 and random.random() < 0.15:
            fillers = ["Interestingly,", "In fact,", "Notably,", "Admittedly,"]
            sent = f"{random.choice(fillers)} {sent[0].lower()}{sent[1:]}"
        result.append(sent)
    return result
