"""
humanizer.py
------------
Humanization Engine (Controlled Naturalization) for DocuGuard+.

Provides two strategies:

1. **Rule-Based** — deterministic transformations (sentence splitting,
   contraction injection, transition variation, length perturbation).
   Works fully offline with zero latency.

2. **LLM-Based** — production-grade multi-pass pipeline that:
   a) Splits text into paragraphs & rewrites each via Ollama (chat API)
   b) Runs our AI detector on the rewritten text
   c) If AI probability is still high, performs a *targeted* second pass
      focusing specifically on the signals the detector flagged
   d) Applies a final rule-based polish for contraction/transition cleanup

After humanization the AI Detection Engine is re-run automatically to produce
a "Before vs. After" probability comparison.
"""

from __future__ import annotations

import logging
import os
import random
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import requests

from utils.text_utils import split_sentences

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("DEFAULT_OLLAMA_MODEL", "mistral")

# Ollama generation parameters – tuned for varied, creative output
_GENERATION_OPTIONS: dict[str, Any] = {
    "temperature": 0.85,         # Higher → more diverse word choice
    "top_p": 0.92,               # Nucleus sampling
    "top_k": 60,
    "repeat_penalty": 1.25,      # Discourage verbatim repetitions
    "num_predict": 2048,         # Max tokens per generation
}

# Thresholds
_TARGET_AI_PROB = 0.35           # Stop iterating once AI prob is below this
_MAX_PASSES = 2                  # Maximum LLM rewriting passes

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


# Type alias for an optional progress callback (stage_name, pct)
ProgressCallback = Optional[Callable[[str, float], None]]


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
        progress_cb: ProgressCallback = None,
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
        progress_cb :
            Optional ``(stage_name: str, progress: float)`` callback for UI
            progress bars.  *progress* goes from 0.0 → 1.0.

        Returns
        -------
        HumanizationResult
            Contains original, rewritten text, and a summary of changes.
        """
        if strategy == "llm":
            return self._llm_humanize(text, model, progress_cb)
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
    # LLM-based rewriting — multi-pass production pipeline
    # ------------------------------------------------------------------

    def _llm_humanize(
        self, text: str, model: str, progress_cb: ProgressCallback = None,
    ) -> HumanizationResult:
        """Production-grade LLM humanization pipeline.

        Pipeline stages
        ----------------
        1. **Paragraph-level rewriting** — split text into paragraphs; rewrite
           each one independently so the LLM can focus on quality.
        2. **AI-detection gate** — run our own detector on the reassembled text.
           If AI probability < target, stop early.
        3. **Targeted second pass** — feed the detector's *specific* flagged
           signals back to the LLM with instructions to fix them.
        4. **Rule-based polish** — apply contractions, transition cleanup, and
           vocabulary simplification as a final pass.
        """
        changes: List[str] = []
        _cb = progress_cb or (lambda *_: None)

        # ── Stage 1: paragraph-level rewriting ──────────────────────────
        _cb("Splitting into paragraphs …", 0.0)
        paragraphs = _split_into_paragraphs(text)
        log.info("LLM humanize: %d paragraphs", len(paragraphs))

        rewritten_parts: List[str] = []
        for idx, para in enumerate(paragraphs):
            _cb(f"Rewriting paragraph {idx + 1}/{len(paragraphs)} …",
                (idx / len(paragraphs)) * 0.55)

            if len(para.split()) < 5:
                # Too short — pass through
                rewritten_parts.append(para)
                continue

            rewritten = self._rewrite_paragraph(para, model, pass_number=1)
            rewritten_parts.append(rewritten)

        draft_1 = "\n\n".join(rewritten_parts)
        changes.append(f"Pass 1: rewrote {len(paragraphs)} paragraphs via {model}")

        # ── Stage 2: AI-detection gate ──────────────────────────────────
        _cb("Evaluating AI probability …", 0.60)
        try:
            from core.ai_detector import AIDetector
            detector = AIDetector()
            gate_result = detector.detect(draft_1)
            ai_prob = gate_result.ai_probability
            flagged = gate_result.explanations
            features = gate_result.features
            changes.append(
                f"Mid-pipeline AI prob: {ai_prob:.0%} "
                f"(perplexity {gate_result.perplexity_result.perplexity:.1f})"
            )
            log.info("Post-pass-1 AI prob: %.2f", ai_prob)
        except Exception as exc:
            log.warning("AI detection gate failed: %s", exc)
            ai_prob = 1.0  # Assume worst case → trigger pass 2
            flagged = []
            features = {}

        # ── Stage 3: targeted second pass (only if still flagged) ───────
        if ai_prob >= _TARGET_AI_PROB and _MAX_PASSES >= 2:
            _cb("Targeted second pass …", 0.65)

            # Build a diagnosis string the LLM can act on
            diagnosis = _build_diagnosis(ai_prob, features, flagged)
            log.info("Pass 2 diagnosis:\n%s", diagnosis)

            # Rewrite the entire draft again with specific fixes
            pass2 = self._targeted_rewrite(draft_1, model, diagnosis)
            if pass2 and len(pass2.split()) > len(draft_1.split()) * 0.4:
                draft_1 = pass2
                changes.append("Pass 2: targeted fixes for flagged AI signals")
            else:
                changes.append("Pass 2: skipped (LLM output too short or empty)")

        # ── Stage 4: rule-based polish ──────────────────────────────────
        _cb("Final rule-based polish …", 0.85)
        polished = self._rule_polish(draft_1)
        changes.append("Final polish: contractions, transitions, vocabulary")

        _cb("Done", 1.0)

        return HumanizationResult(
            original_text=text,
            humanized_text=polished,
            strategy="llm",
            changes_summary=changes,
        )

    # ------------------------------------------------------------------
    # Ollama API helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ollama_chat(
        model: str,
        system_prompt: str,
        user_prompt: str,
        timeout: int = 300,
    ) -> str:
        """Call Ollama /api/chat with system + user messages."""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": _GENERATION_OPTIONS,
        }
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()

    # ------------------------------------------------------------------
    # Pass 1: paragraph-level rewriting
    # ------------------------------------------------------------------

    _SYSTEM_PROMPT_P1 = textwrap.dedent("""\
        You are an expert academic writing coach. Your ONLY job is to
        rewrite user-provided text so it reads as authentically
        human-written while keeping every factual claim intact.

        AI-generated text has these detectable patterns that you MUST break:
        • Uniform sentence lengths → vary dramatically (5-word punchy + 30-word complex)
        • Transition-word overuse (Furthermore/Moreover/Additionally) → drop or replace
        • Predictable Subject-Verb-Object structure every sentence → vary syntax
        • Low vocabulary diversity  → use synonyms, simpler words, slang where apt
        • Overly formal register   → add contractions, asides, rhetorical questions

        Rules:
        1. Output ONLY the rewritten text.  No preamble, commentary, or labels.
        2. Preserve all factual content and nuance.
        3. Keep roughly the same length (within ±20 %).
        4. Never add disclaimers like "Here is the rewritten text".
    """)

    def _rewrite_paragraph(self, paragraph: str, model: str, pass_number: int = 1) -> str:
        """Rewrite a single paragraph via Ollama chat API."""
        try:
            result = self._ollama_chat(
                model=model,
                system_prompt=self._SYSTEM_PROMPT_P1,
                user_prompt=paragraph,
                timeout=max(120, len(paragraph.split()) * 2),
            )
            # Validate: not empty, not absurdly short
            if result and len(result.split()) > len(paragraph.split()) * 0.3:
                return result
            log.warning("Pass %d: LLM output too short, keeping original", pass_number)
            return paragraph
        except Exception as exc:
            log.warning("Pass %d LLM call failed: %s", pass_number, exc)
            return paragraph

    # ------------------------------------------------------------------
    # Pass 2: targeted rewrite based on detector diagnosis
    # ------------------------------------------------------------------

    _SYSTEM_PROMPT_P2 = textwrap.dedent("""\
        You are an AI-text remediation specialist. The user will give you:
        1. A DIAGNOSIS section listing the exact AI signals detected.
        2. The TEXT to fix.

        Your job: rewrite the text to specifically address EVERY flagged
        issue in the diagnosis while keeping ALL facts intact.

        Guidelines:
        • If "low burstiness" → make sentence lengths dramatically uneven
        • If "transition overuse" → remove or replace every transition starter
        • If "low perplexity" → introduce surprises: unusual word choices,
          rhetorical questions, parenthetical asides, sentence fragments
        • If "uniformity" → break parallel structures, vary paragraph lengths
        • If "vocabulary" → swap academic jargon for plain language

        Output ONLY the rewritten text.  No commentary.
    """)

    def _targeted_rewrite(self, text: str, model: str, diagnosis: str) -> str:
        """Run a targeted second-pass rewrite guided by detector diagnosis."""
        user_msg = f"DIAGNOSIS:\n{diagnosis}\n\n---\n\nTEXT:\n{text}"
        try:
            result = self._ollama_chat(
                model=model,
                system_prompt=self._SYSTEM_PROMPT_P2,
                user_prompt=user_msg,
                timeout=max(180, len(text.split()) * 2),
            )
            return result
        except Exception as exc:
            log.warning("Pass 2 LLM call failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Final rule-based polish (lightweight)
    # ------------------------------------------------------------------

    def _rule_polish(self, text: str) -> str:
        """Apply non-LLM transforms as a final cleanup."""
        sentences = split_sentences(text)
        polished: List[str] = []
        for sent in sentences:
            s, _ = _inject_contractions(sent)
            s, _ = _vary_transitions(s)
            s, _ = _drop_leading_transition(s)
            polished.append(s)
        result = " ".join(polished)
        result, _ = _simplify_vocabulary(result)
        return result


# ---------------------------------------------------------------------------
# Paragraph & diagnosis helpers
# ---------------------------------------------------------------------------

def _split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs, keeping minimum ~3 sentences each.

    If the text has no blank-line breaks, split at every 3-5 sentences so
    each chunk is manageable for the LLM.
    """
    # Try natural paragraph breaks first
    raw_paras = re.split(r"\n\s*\n", text.strip())
    raw_paras = [p.strip() for p in raw_paras if p.strip()]

    if len(raw_paras) >= 2:
        # Merge very short paragraphs together
        merged: List[str] = []
        for p in raw_paras:
            if merged and len(merged[-1].split()) < 30:
                merged[-1] = merged[-1] + " " + p
            else:
                merged.append(p)
        return merged

    # Fallback: split by sentences into chunks of 3-5
    sentences = split_sentences(text)
    if len(sentences) <= 5:
        return [text]

    paragraphs: List[str] = []
    chunk: List[str] = []
    for sent in sentences:
        chunk.append(sent)
        if len(chunk) >= random.randint(3, 5):
            paragraphs.append(" ".join(chunk))
            chunk = []
    if chunk:
        paragraphs.append(" ".join(chunk))
    return paragraphs


def _build_diagnosis(
    ai_prob: float,
    features: Dict[str, Any],
    explanations: List[str],
) -> str:
    """Convert detector output into an actionable diagnosis for pass 2."""
    lines: List[str] = [f"Current AI probability: {ai_prob:.0%}"]

    burstiness = features.get("burstiness", 0)
    if burstiness < 0.45:
        lines.append(
            f"• LOW BURSTINESS ({burstiness:.2f}): Sentence lengths are too "
            "uniform. Mix very short (3-6 words) and long (25+) sentences."
        )

    ttr = features.get("type_token_ratio", 1.0)
    if ttr < 0.55:
        lines.append(
            f"• LOW VOCABULARY DIVERSITY (TTR={ttr:.2f}): Too many repeated "
            "words. Use more synonyms and varied phrasing."
        )

    transition_freq = features.get("transition_word_freq", 0)
    if transition_freq > 0.03:
        lines.append(
            f"• HIGH TRANSITION FREQUENCY ({transition_freq:.3f}): Too many "
            "transition starters. Drop most of them entirely."
        )

    starter_div = features.get("sentence_starter_diversity", 1.0)
    if starter_div < 0.6:
        lines.append(
            f"• LOW STARTER DIVERSITY ({starter_div:.2f}): Sentences begin "
            "with the same words. Vary your openings."
        )

    mean_sent_len = features.get("mean_sentence_length", 0)
    std_sent_len = features.get("std_sentence_length", 0)
    if std_sent_len < 5:
        lines.append(
            f"• UNIFORM SENTENCE LENGTH (mean={mean_sent_len:.0f}, "
            f"std={std_sent_len:.1f}): Drastically vary lengths."
        )

    # Include any detector explanations that mention specific issues
    for exp in explanations:
        if any(kw in exp.lower() for kw in ("uniform", "burstiness", "transition",
                                              "perplexity", "repetit")):
            lines.append(f"• Detector note: {exp}")

    return "\n".join(lines)


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
