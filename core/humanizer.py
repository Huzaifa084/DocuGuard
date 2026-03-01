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

        for i, sent in enumerate(sentences):
            s = sent

            # 1. ALWAYS replace overused transitions (high probability)
            s, t_changed = _vary_transitions(s)
            if t_changed:
                changes.append("Varied transition words")

            # 2. Drop leading transitions entirely (every 2nd-3rd sentence)
            s, dropped = _drop_leading_transition(s)
            if dropped:
                changes.append("Removed unnecessary transition starters")

            # 3. ALWAYS inject contractions where possible
            s, c_changed = _inject_contractions(s)
            if c_changed:
                changes.append("Injected contractions")

            # 4. Split long sentences more aggressively (> 20 words)
            if len(s.split()) > 20 and random.random() < 0.6:
                parts = _split_long_sentence(s)
                if len(parts) > 1:
                    new_sentences.extend(parts)
                    changes.append("Split long sentences")
                    continue

            # 5. Merge short consecutive sentences more often
            if (
                new_sentences
                and len(new_sentences[-1].split()) < 10
                and len(s.split()) < 10
                and random.random() < 0.45
            ):
                merged = new_sentences.pop().rstrip(".")
                connectors = [", and ", " — ", "; ", ", which means "]
                conn = random.choice(connectors)
                s = f"{merged}{conn}{s[0].lower()}{s[1:]}"
                changes.append("Merged short sentences")

            # 6. Reorder clauses in sentences with commas
            s, reordered = _reorder_clauses(s)
            if reordered:
                changes.append("Reordered sentence clauses")

            # 7. Add parenthetical asides occasionally
            if len(s.split()) > 12 and random.random() < 0.2:
                s = _add_parenthetical(s)
                changes.append("Added parenthetical asides")

            # 8. Vary sentence starters — invert some to start differently
            if i > 0 and random.random() < 0.25:
                s = _vary_sentence_start(s)
                changes.append("Varied sentence openings")

            new_sentences.append(s)

        # 9. Length perturbation
        new_sentences = _perturb_lengths(new_sentences)
        if len(new_sentences) != len(sentences):
            changes.append("Adjusted sentence lengths for variability")

        # 10. Replace overused academic vocabulary
        humanized = " ".join(new_sentences)
        humanized, vocab_changed = _simplify_vocabulary(humanized)
        if vocab_changed:
            changes.append("Simplified academic vocabulary")

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
            "You are an expert academic writing coach who specialises in making "
            "text sound authentically human. AI-generated text has telltale "
            "patterns: uniform sentence lengths, repetitive transition words "
            "(Furthermore/Moreover/Additionally), predictable structure, low "
            "vocabulary variation, and overly formal tone.\n\n"
            "Rewrite the text below to eliminate ALL of these AI patterns while "
            "preserving every factual claim and the overall meaning.\n\n"
            "MANDATORY changes:\n"
            "1. VARY sentence lengths dramatically — mix very short punchy "
            "sentences (3-6 words) with longer complex ones (25+ words). "
            "Human writing has high 'burstiness'.\n"
            "2. ELIMINATE generic transitions (Furthermore, Moreover, "
            "Additionally, Consequently). Replace with natural connectors, "
            "or simply drop them — real writers don't start every sentence "
            "with a transition.\n"
            "3. RESTRUCTURE paragraphs — combine some sentences, split others, "
            "change the order of clauses within sentences.\n"
            "4. ADD human touches — occasional rhetorical questions, "
            "parenthetical asides (like this one), and informal phrasing "
            "where appropriate.\n"
            "5. USE contractions naturally (it's, don't, can't, won't).\n"
            "6. VARY vocabulary — replace repeated academic words with "
            "synonyms or simpler alternatives.\n"
            "7. BREAK parallel structures — don't let consecutive sentences "
            "follow the same Subject-Verb-Object pattern.\n\n"
            "Return ONLY the rewritten text. No commentary, no preamble, "
            "no 'Here is the rewritten text'. Just the text itself.\n\n"
            f"TEXT:\n{text}"
        )

        # Longer timeout for large texts on CPU-only Ollama
        timeout_secs = max(180, len(text.split()) // 3)

        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=timeout_secs,
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
        if original in lower and random.random() < 0.85:  # Much higher probability
            replacement = random.choice(alternatives)
            pattern = re.compile(re.escape(original), re.IGNORECASE)
            match = pattern.search(sentence)
            if match:
                orig_text = match.group()
                if orig_text[0].isupper():
                    replacement = replacement[0].upper() + replacement[1:]
                sentence = pattern.sub(replacement, sentence, count=1)
                changed = True
    return sentence, changed


def _drop_leading_transition(sentence: str) -> tuple[str, bool]:
    """Remove transition words at the start of a sentence entirely."""
    drop_starters = [
        "furthermore,", "moreover,", "additionally,", "consequently,",
        "in addition,", "as a result,", "also,", "besides,",
        "on top of that,", "beyond that,", "plus,", "what's more,",
        "in fact,", "notably,", "specifically,",
    ]
    lower = sentence.strip().lower()
    if random.random() < 0.5:  # Drop ~50% of leading transitions
        for starter in drop_starters:
            if lower.startswith(starter):
                rest = sentence.strip()[len(starter):].strip()
                if rest:
                    return rest[0].upper() + rest[1:], True
    return sentence, False


def _reorder_clauses(sentence: str) -> tuple[str, bool]:
    """Swap main and subordinate clauses when a comma separates them."""
    if random.random() > 0.3:
        return sentence, False
    # Find sentences with a dependent clause at the start
    parts = sentence.split(", ", 1)
    if len(parts) == 2 and 5 < len(parts[0].split()) < 15 and len(parts[1].split()) > 4:
        second = parts[1].rstrip(".")
        first = parts[0]
        # Only reorder if second part doesn't start with conjunction
        if not second.split()[0].lower() in {"and", "but", "or", "which", "who", "that"}:
            reordered = f"{second[0].upper()}{second[1:]}, {first[0].lower()}{first[1:]}."
            return reordered, True
    return sentence, False


def _add_parenthetical(sentence: str) -> str:
    """Insert a parenthetical aside into the middle of a sentence."""
    asides = [
        "(at least in theory)",
        "(and this matters)",
        "(to some extent)",
        "(perhaps unsurprisingly)",
        "(which is worth noting)",
        "(broadly speaking)",
        "(in most cases)",
        "(for better or worse)",
    ]
    words = sentence.split()
    if len(words) < 8:
        return sentence
    mid = len(words) // 2
    # Find a natural break point near the middle
    for offset in range(min(4, mid)):
        for pos in [mid + offset, mid - offset]:
            if 0 < pos < len(words) and words[pos - 1].endswith(","):
                words.insert(pos, random.choice(asides))
                return " ".join(words)
    # Fallback: insert after a reasonable word
    words.insert(mid, random.choice(asides))
    return " ".join(words)


def _vary_sentence_start(sentence: str) -> str:
    """Rephrase the beginning of a sentence for variety."""
    starters = [
        "Interestingly, ", "In practice, ", "From this perspective, ",
        "It turns out that ", "What stands out is that ",
        "Looking at the data, ", "To put it simply, ",
        "In many ways, ",
    ]
    # Only if sentence starts with a typical subject (The, This, These, It, A/An)
    first_word = sentence.split()[0] if sentence.split() else ""
    if first_word.lower() in {"the", "this", "these", "it", "a", "an", "such"}:
        return random.choice(starters) + sentence[0].lower() + sentence[1:]
    return sentence


def _simplify_vocabulary(text: str) -> tuple[str, bool]:
    """Replace overused academic/AI words with simpler alternatives."""
    replacements = {
        "utilize": "use", "utilizes": "uses", "utilized": "used",
        "utilizing": "using", "utilization": "use",
        "facilitate": "help", "facilitates": "helps",
        "demonstrate": "show", "demonstrates": "shows",
        "demonstrated": "showed",
        "implement": "put in place", "implementation": "rollout",
        "leverage": "use", "leveraging": "using",
        "numerous": "many", "substantial": "large",
        "significant": "major", "significantly": "greatly",
        "unprecedented": "remarkable",
        "comprehensive": "thorough", "fundamental": "core",
        "fundamentally": "at its core",
        "enhancing": "improving", "enhanced": "improved",
        "pivotal": "key", "crucial": "critical",
        "paradigm": "approach", "methodology": "method",
        "innovative": "new", "robust": "strong",
        "optimal": "best", "streamline": "simplify",
        "landscape": "field", "realm": "area",
        "underscore": "highlight", "underscores": "highlights",
        "endeavor": "effort", "endeavors": "efforts",
        "multifaceted": "complex",
    }
    changed = False
    for word, simple in replacements.items():
        pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
        if pattern.search(text) and random.random() < 0.75:
            # Capture `simple` via default arg to avoid closure-over-loop-variable bug
            def _match_case(m, _s=simple):
                orig = m.group()
                if orig[0].isupper():
                    return _s[0].upper() + _s[1:]
                return _s
            text = pattern.sub(_match_case, text, count=1)
            changed = True
    return text, changed


def _inject_contractions(sentence: str) -> tuple[str, bool]:
    changed = False
    for full, contracted in _CONTRACTION_MAP.items():
        if full.lower() in sentence.lower() and random.random() < 0.8:
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
