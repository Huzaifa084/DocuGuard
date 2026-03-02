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
def _get_ollama_url() -> str:
    """Return the current Ollama URL (supports dynamic Colab tunnel URLs)."""
    return os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")

DEFAULT_MODEL = os.environ.get("DEFAULT_OLLAMA_MODEL", "mistral")

# Hard word-count limit — keeps LLM processing tractable
MAX_WORD_LIMIT: int = 8000

# Ollama generation parameters – tuned for varied, creative output
_GENERATION_OPTIONS: dict[str, Any] = {
    "temperature": 0.85,         # Higher → more diverse word choice
    "top_p": 0.92,               # Nucleus sampling
    "top_k": 60,
    "repeat_penalty": 1.25,      # Discourage verbatim repetitions
    "num_predict": 2048,         # Max tokens per generation
}

# Thresholds
_TARGET_AI_PROB = 0.30           # Stop iterating once AI prob is below this
_MAX_PASSES = 3                  # Maximum LLM rewriting passes

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HumanizationConfig:
    """User-controllable knobs for the LLM humanization pipeline.

    Passed from the Streamlit UI so that the LLM can tailor its rewriting
    to the user's specific needs.
    """
    tone: str = "academic"              # academic | casual | professional | creative
    intensity: str = "balanced"         # light | balanced | aggressive
    domain: str = ""                    # e.g. "computer science", "history"
    preserve_keywords: str = ""         # comma-separated terms to keep verbatim
    custom_instructions: str = ""       # free-form notes for the LLM
    target_ai_prob: float = 0.30        # stop once AI prob drops below this
    max_passes: int = 3                 # cap on LLM rewriting iterations
    output_format: str = "plain"        # plain | markdown


@dataclass
class HumanizationResult:
    """Output of the humanization engine."""
    original_text: str = ""
    humanized_text: str = ""
    strategy: str = ""                  # "rule-based" | "llm"
    changes_summary: List[str] = field(default_factory=list)
    ai_prob_before: Optional[float] = None
    ai_prob_after: Optional[float] = None
    word_count_original: int = 0
    word_count_humanized: int = 0
    passes_used: int = 0
    elapsed_seconds: float = 0.0        # Total processing time


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
        config: Optional[HumanizationConfig] = None,
    ) -> HumanizationResult:
        """Rewrite *text* to improve naturalness.

        Parameters
        ----------
        text :
            The cleaned document text.  Capped at ``MAX_WORD_LIMIT`` words;
            excess text is reattached after processing.
        strategy :
            ``"rule-based"`` (default, offline) or ``"llm"`` (requires Ollama).
        model :
            Ollama model name when *strategy* is ``"llm"``.
        progress_cb :
            Optional ``(stage_name: str, progress: float)`` callback for UI
            progress bars.  *progress* goes from 0.0 → 1.0.
        config :
            ``HumanizationConfig`` with user preferences (tone, intensity,
            domain, custom instructions, etc.).  If *None*, defaults apply.

        Returns
        -------
        HumanizationResult
            Contains original, rewritten text, and a summary of changes.
        """
        cfg = config or HumanizationConfig()

        # ── Word-count enforcement ──────────────────────────────────────
        words = text.split()
        overflow = ""
        if len(words) > MAX_WORD_LIMIT:
            overflow = " ".join(words[MAX_WORD_LIMIT:])
            text = " ".join(words[:MAX_WORD_LIMIT])
            log.info(
                "Input truncated to %d words (overflow: %d words kept as-is)",
                MAX_WORD_LIMIT, len(overflow.split()),
            )

        if strategy == "llm":
            result = self._llm_humanize(text, model, progress_cb, cfg)
        else:
            result = self._rule_based_humanize(text)

        # Reattach overflow (un-processed tail) if any
        if overflow:
            result.humanized_text = result.humanized_text.rstrip() + "\n\n" + overflow
            result.changes_summary.append(
                f"Note: only the first {MAX_WORD_LIMIT:,} words were processed; "
                f"{len(overflow.split()):,} trailing words appended unchanged."
            )

        result.word_count_original = len(result.original_text.split())
        result.word_count_humanized = len(result.humanized_text.split())
        return result

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
        self,
        text: str,
        model: str,
        progress_cb: ProgressCallback = None,
        config: Optional[HumanizationConfig] = None,
    ) -> HumanizationResult:
        """Production-grade LLM humanization pipeline.

        Pipeline
        --------
        0. **Pre-analysis** — run AI detector to identify *what makes the text
           AI-like* before any rewriting begins.
        1. **Context-aware paragraph rewriting** — paragraphs are rewritten one
           at a time, but the LLM receives the *previous* and *next* paragraph
           as read-only context for coherence.
        2. **AI-detection gate** — check AI probability after the pass.
        3. **Sentence-level targeted fixes** — individual high-AI-probability
           sentences identified by the detector are rewritten individually.
        4. **Iterative loop** (up to ``config.max_passes``) — if AI probability
           is still above the target, run another full-pass with the latest
           diagnosis.
        5. **Rule-based polish** — contractions, transition cleanup, vocabulary
           simplification.

        The entire pipeline processes **all text** (up to the word-count cap).
        """
        cfg = config or HumanizationConfig()
        target_prob = cfg.target_ai_prob
        max_passes = min(cfg.max_passes, _MAX_PASSES)
        changes: List[str] = []
        _cb = progress_cb or (lambda *_: None)
        passes_used = 0

        # ── Stage 0: Pre-analysis ───────────────────────────────────────
        _cb("Analysing text before rewriting …", 0.0)
        pre_features: Dict[str, Any] = {}
        pre_explanations: List[str] = []
        pre_ai_prob = 1.0
        try:
            from core.ai_detector import AIDetector
            detector = AIDetector()
            pre_result = detector.detect(text)
            pre_ai_prob = pre_result.ai_probability
            pre_features = pre_result.features
            pre_explanations = pre_result.explanations
            changes.append(
                f"Pre-analysis: AI prob {pre_ai_prob:.0%}, "
                f"perplexity {pre_result.perplexity_result.perplexity:.1f}"
            )
        except Exception as exc:
            log.warning("Pre-analysis failed: %s", exc)

        # Build dynamic system prompt incorporating user preferences
        sys_prompt = self._build_system_prompt(cfg, pre_features, pre_explanations)

        # ── Stage 1: context-aware paragraph rewriting ──────────────────
        _cb("Splitting into paragraphs …", 0.03)
        paragraphs = _split_into_paragraphs(text)
        n_paras = len(paragraphs)
        log.info("LLM humanize: %d paragraphs, model=%s", n_paras, model)

        draft = self._rewrite_all_paragraphs(
            paragraphs, model, sys_prompt, _cb,
            progress_start=0.05, progress_end=0.45,
        )
        passes_used += 1
        changes.append(f"Pass 1: rewrote {n_paras} paragraphs via {model}")

        # ── Iterative AI-detection gate + targeted passes ───────────────
        for pass_num in range(2, max_passes + 1):
            # Check AI probability
            gate_pct = 0.45 + (pass_num - 2) * 0.15
            _cb(f"AI detection gate (pass {pass_num - 1} result) …", gate_pct)
            ai_prob, features, explanations = self._run_detection_gate(draft)
            changes.append(
                f"Post-pass-{pass_num - 1} AI prob: {ai_prob:.0%}"
            )
            log.info("Post-pass-%d AI prob: %.2f", pass_num - 1, ai_prob)

            if ai_prob < target_prob:
                changes.append(f"AI prob {ai_prob:.0%} < target {target_prob:.0%} — stopping early")
                break

            # Build diagnosis from latest detection
            diagnosis = _build_diagnosis(ai_prob, features, explanations)

            # Sentence-level targeted fixes for the worst offenders
            _cb(f"Sentence-level fixes (pass {pass_num}) …", gate_pct + 0.03)
            draft, n_fixed = self._fix_worst_sentences(
                draft, model, cfg, diagnosis,
            )
            if n_fixed:
                changes.append(f"Pass {pass_num}: fixed {n_fixed} high-AI sentences")

            # Full targeted rewrite if still above threshold
            ai_prob2, features2, explanations2 = self._run_detection_gate(draft)
            if ai_prob2 >= target_prob:
                _cb(f"Full targeted rewrite (pass {pass_num}) …", gate_pct + 0.07)
                diagnosis2 = _build_diagnosis(ai_prob2, features2, explanations2)
                rewritten = self._targeted_rewrite(draft, model, diagnosis2, cfg)
                if rewritten and len(rewritten.split()) > len(draft.split()) * 0.4:
                    draft = rewritten
                    changes.append(f"Pass {pass_num}: full targeted rewrite")

            passes_used += 1
        else:
            # Loop completed without meeting target
            final_ai_prob, _, _ = self._run_detection_gate(draft)
            if final_ai_prob >= target_prob:
                changes.append(
                    f"Max passes ({max_passes}) reached. Final AI prob: {final_ai_prob:.0%} "
                    f"(target was {target_prob:.0%})"
                )

        # ── Final rule-based polish ─────────────────────────────────────
        _cb("Final rule-based polish …", 0.90)
        polished = self._rule_polish(draft)
        changes.append("Final polish: contractions, transitions, vocabulary")

        _cb("Done", 1.0)

        return HumanizationResult(
            original_text=text,
            humanized_text=polished,
            strategy="llm",
            changes_summary=changes,
            passes_used=passes_used,
        )

    # ------------------------------------------------------------------
    # Dynamic prompt builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_system_prompt(
        cfg: HumanizationConfig,
        pre_features: Dict[str, Any],
        pre_explanations: List[str],
    ) -> str:
        """Construct a system prompt tailored to user config + pre-analysis."""

        # ── Tone mapping ────────────────────────────────────────────────
        tone_guide = {
            "academic": (
                "Write in a scholarly but readable style.  Use field-specific "
                "terminology where appropriate, but prefer clarity over jargon. "
                "Contractions are acceptable when they improve flow."
            ),
            "casual": (
                "Write informally — like a knowledgeable friend explaining "
                "something over coffee.  Use contractions, short punchy "
                "sentences, and occasional humor or rhetorical questions."
            ),
            "professional": (
                "Write in a polished business-appropriate tone.  Be concise, "
                "direct, and authoritative.  Use contractions sparingly."
            ),
            "creative": (
                "Write with flair — vivid verbs, varied rhythm, unexpected "
                "word choices.  Let personality shine through while "
                "preserving factual accuracy."
            ),
        }
        tone_text = tone_guide.get(cfg.tone, tone_guide["academic"])

        # ── Intensity mapping ───────────────────────────────────────────
        intensity_guide = {
            "light": (
                "Make minimal changes.  Preserve the author's original "
                "voice as much as possible; only fix the most obvious "
                "AI patterns."
            ),
            "balanced": (
                "Rewrite substantially but keep the core structure. "
                "Break AI patterns while preserving factual content and "
                "the general argument flow."
            ),
            "aggressive": (
                "Rewrite boldly — restructure paragraphs, reorder "
                "arguments, vary sentence lengths dramatically, introduce "
                "surprising word choices.  The output should be virtually "
                "unrecognisable as AI-generated."
            ),
        }
        intensity_text = intensity_guide.get(cfg.intensity, intensity_guide["balanced"])

        # ── Pre-analysis insights ───────────────────────────────────────
        issues: List[str] = []
        burstiness = pre_features.get("burstiness", 0.5)
        if burstiness < 0.45:
            issues.append(f"LOW BURSTINESS ({burstiness:.2f}) — sentence lengths too uniform")
        ttr = pre_features.get("type_token_ratio", 0.6)
        if ttr < 0.55:
            issues.append(f"LOW VOCABULARY DIVERSITY (TTR={ttr:.2f})")
        transition = pre_features.get("transition_word_ratio", 0)
        if transition > 0.02:
            issues.append(f"TRANSITION OVERUSE ({transition:.3f})")
        passive = pre_features.get("passive_voice_ratio", 0)
        if passive > 0.25:
            issues.append(f"HIGH PASSIVE VOICE ({passive:.2f})")
        starter_div = pre_features.get("sentence_starter_diversity", 1)
        if starter_div < 0.6:
            issues.append(f"LOW STARTER DIVERSITY ({starter_div:.2f})")

        issues_block = ""
        if issues:
            bullet_list = "\n".join(f"  • {i}" for i in issues)
            issues_block = (
                f"\n\nPRE-ANALYSIS — the following AI signals were detected "
                f"in the ORIGINAL text.  You MUST specifically fix these:\n"
                f"{bullet_list}"
            )

        # ── Domain / keywords / custom ──────────────────────────────────
        extras: List[str] = []
        if cfg.domain:
            extras.append(f"The text is about **{cfg.domain}**.  Use domain-appropriate vocabulary.")
        if cfg.preserve_keywords:
            extras.append(
                f"These terms MUST appear verbatim (do not paraphrase them): "
                f"{cfg.preserve_keywords}"
            )
        if cfg.custom_instructions:
            extras.append(f"Additional user instructions: {cfg.custom_instructions}")
        extras_block = "\n".join(extras)
        if extras_block:
            extras_block = "\n\n" + extras_block

        return textwrap.dedent(f"""\
            You are an expert writing coach specialising in making text
            read as authentically human-written.  Your ONLY job is to
            rewrite the user-provided text.

            TONE: {tone_text}

            INTENSITY: {intensity_text}

            AI-generated text has these detectable patterns — BREAK them:
            • Uniform sentence lengths → vary dramatically (3-word punchy + 30-word complex)
            • Transition-word overuse (Furthermore/Moreover/Additionally) → remove or replace
            • Predictable Subject-Verb-Object every sentence → vary syntax (inversions, fragments, questions)
            • Low vocabulary diversity → synonyms, simpler words, domain-appropriate terms
            • Overly formal register → contractions, asides, rhetorical questions where appropriate
            • Uniform paragraph lengths → vary: one paragraph 2 sentences, next 6 sentences
            • Overuse of passive voice → prefer active constructions
            {issues_block}

            HARD RULES:
            1. Output ONLY the rewritten text.  No preamble, commentary, labels, or meta-text.
            2. Preserve ALL factual content, claims, and nuance.
            3. Keep roughly the same length (within ±20 %).
            4. Never begin with "Here is" / "Sure" / "Certainly" or similar.
            5. Do NOT add transition words (Moreover, Furthermore, Additionally, etc.).
            {extras_block}

            {"OUTPUT FORMAT: Use clean Markdown formatting throughout the output. Rules:" + chr(10) + "            - Use **bold** for key terms and emphasis" + chr(10) + "            - Use *italics* for nuanced or softer emphasis" + chr(10) + "            - Use bullet lists (- item) when listing related points" + chr(10) + "            - Use ## and ### headings to structure major sections if the text has clear sections" + chr(10) + "            - Use > blockquotes for important callouts or definitions" + chr(10) + "            - Keep paragraphs well-separated with blank lines" + chr(10) + "            - Make the output visually scannable and reader-friendly" + chr(10) + "            - Do NOT wrap the entire output in a code block" if cfg.output_format == "markdown" else "OUTPUT FORMAT: Plain text only — no Markdown formatting, no special characters for formatting."}
        """)

    # ------------------------------------------------------------------
    # Ollama API helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ollama_chat(
        model: str,
        system_prompt: str,
        user_prompt: str,
        timeout: int = 300,
        max_retries: int = 3,
    ) -> str:
        """Call Ollama /api/chat with system + user messages.

        Includes exponential back-off retry for transient failures
        (connection errors, 5xx from Ollama, timeouts).
        """
        import time

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": _GENERATION_OPTIONS,
        }

        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(
                    f"{_get_ollama_url()}/api/chat", json=payload, timeout=timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("message", {}).get("content", "").strip()
            except (requests.ConnectionError, requests.Timeout,
                    requests.HTTPError) as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = 2 ** attempt  # 2s, 4s
                    log.warning(
                        "Ollama call attempt %d/%d failed (%s), retrying in %ds …",
                        attempt, max_retries, exc, wait,
                    )
                    time.sleep(wait)

        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Context-aware paragraph rewriting
    # ------------------------------------------------------------------

    def _rewrite_all_paragraphs(
        self,
        paragraphs: List[str],
        model: str,
        system_prompt: str,
        progress_cb: Callable,
        progress_start: float = 0.05,
        progress_end: float = 0.45,
    ) -> str:
        """Rewrite every paragraph, passing neighbouring context."""
        n = len(paragraphs)
        rewritten: List[str] = []

        for idx, para in enumerate(paragraphs):
            pct = progress_start + (idx / max(n, 1)) * (progress_end - progress_start)
            progress_cb(f"Rewriting paragraph {idx + 1}/{n} …", pct)

            if len(para.split()) < 5:
                rewritten.append(para)
                continue

            # Build context window: previous (REWRITTEN) + next (original) paragraph
            ctx_parts: List[str] = []
            if idx > 0:
                # Use the ALREADY REWRITTEN previous paragraph for coherence
                ctx_parts.append(f"[PREVIOUS PARAGRAPH — for context only, do NOT rewrite]\n{rewritten[idx - 1]}")
            ctx_parts.append(f"[REWRITE THIS PARAGRAPH]\n{para}")
            if idx < n - 1:
                ctx_parts.append(f"[NEXT PARAGRAPH — for context only, do NOT rewrite]\n{paragraphs[idx + 1]}")

            user_msg = "\n\n".join(ctx_parts)

            try:
                result = self._ollama_chat(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_msg,
                    timeout=max(120, len(para.split()) * 3),
                )
                # Validate output: check length bounds and meta-commentary
                result_words = len(result.split()) if result else 0
                para_words = len(para.split())
                if not result or result_words < para_words * 0.3:
                    log.warning("Paragraph %d: LLM output too short (%d vs %d), keeping original", idx, result_words, para_words)
                    rewritten.append(para)
                elif result_words > para_words * 3:
                    log.warning("Paragraph %d: LLM output too long (%d vs %d), keeping original", idx, result_words, para_words)
                    rewritten.append(para)
                else:
                    # Strip meta-commentary prefixes
                    result_lower = result.lower().lstrip()
                    if result_lower.startswith(("here", "sure", "certainly", "i've", "i have", "the rewritten")):
                        # Find first newline or period and skip the preamble
                        for sep in ("\n\n", "\n", ". "):
                            if sep in result:
                                result = result.split(sep, 1)[1]
                                break
                    # Strip any accidentally included context paragraphs
                    result = self._strip_context_leakage(result, paragraphs, rewritten, idx)
                    rewritten.append(result)
            except Exception as exc:
                log.warning("Paragraph %d rewrite failed: %s", idx, exc)
                rewritten.append(para)

        return "\n\n".join(rewritten)

    @staticmethod
    def _strip_context_leakage(
        result: str, paragraphs: List[str], rewritten: List[str], current_idx: int,
    ) -> str:
        """Remove accidentally repeated context paragraphs from LLM output."""
        # Remove marker lines
        lines = result.strip().split("\n")
        filtered = [
            ln for ln in lines
            if not ln.strip().startswith("[PREVIOUS PARAGRAPH")
            and not ln.strip().startswith("[NEXT PARAGRAPH")
            and not ln.strip().startswith("[REWRITE THIS")
        ]
        result = "\n".join(filtered).strip()

        # Remove actual context paragraph content if accidentally included
        if current_idx > 0 and len(rewritten) > current_idx - 1:
            prev_para = rewritten[current_idx - 1]
            if prev_para in result:
                result = result.replace(prev_para, "").strip()
        if current_idx < len(paragraphs) - 1:
            next_para = paragraphs[current_idx + 1]
            if next_para in result:
                result = result.replace(next_para, "").strip()

        return result

    # ------------------------------------------------------------------
    # AI-detection gate helper
    # ------------------------------------------------------------------

    @staticmethod
    def _run_detection_gate(text: str) -> tuple:
        """Run AI detection and return (ai_prob, features, explanations)."""
        try:
            from core.ai_detector import AIDetector
            detector = AIDetector()
            result = detector.detect(text)
            return result.ai_probability, result.features, result.explanations
        except Exception as exc:
            log.warning("Detection gate failed: %s", exc)
            return 1.0, {}, []

    # ------------------------------------------------------------------
    # Sentence-level targeted fixes
    # ------------------------------------------------------------------

    def _fix_worst_sentences(
        self,
        text: str,
        model: str,
        cfg: HumanizationConfig,
        diagnosis: str,
        max_fixes: int = 10,
    ) -> tuple[str, int]:
        """Identify and rewrite the most "AI-like" individual sentences.

        Uses a lightweight per-sentence AI probability estimate (perplexity +
        feature extraction on the sentence in context) to find the worst
        offenders, then rewrites them one by one.

        Preserves original paragraph structure by tracking paragraph boundaries.
        """
        import time

        # Split into paragraphs first to preserve structure
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return text, 0

        # Build a flat list of (paragraph_idx, sentence) tuples
        all_sentences: List[tuple[int, str]] = []
        for para_idx, para in enumerate(paragraphs):
            para_sents = split_sentences(para)
            for sent in para_sents:
                all_sentences.append((para_idx, sent))

        if len(all_sentences) < 3:
            return text, 0

        # Score each sentence by a quick heuristic:
        mean_len = sum(len(s.split()) for _, s in all_sentences) / len(all_sentences)
        transition_starters = {
            "furthermore", "moreover", "additionally", "consequently",
            "nevertheless", "subsequently", "however", "therefore",
            "interestingly", "notably", "importantly", "significantly",
            "essentially", "ultimately", "indeed",
        }
        be_forms = {"am", "is", "are", "was", "were", "been", "being"}

        scored: List[tuple[int, float]] = []  # (flat_index, ai_score)
        for i, (_, sent) in enumerate(all_sentences):
            score = 0.0
            words = sent.split()
            wlen = len(words)
            if not words:
                continue

            # Transition starter
            fw = words[0].lower().rstrip(",")
            if fw in transition_starters:
                score += 0.4

            # Length too close to mean (uniform)
            if abs(wlen - mean_len) < 3:
                score += 0.2

            # Passive voice indicator
            lower_words = [w.lower() for w in words]
            for j in range(len(lower_words) - 1):
                if lower_words[j] in be_forms and j + 1 < len(lower_words):
                    nxt = lower_words[j + 1]
                    if nxt.endswith("ed") or nxt.endswith("en"):
                        score += 0.15
                        break

            # Very long without any punctuation variety
            if wlen > 20 and "," not in sent and ";" not in sent:
                score += 0.15

            scored.append((i, score))

        # Pick the worst offenders
        scored.sort(key=lambda x: x[1], reverse=True)
        to_fix = [(idx, sc) for idx, sc in scored[:max_fixes] if sc >= 0.3]

        if not to_fix:
            return text, 0

        sys_prompt = textwrap.dedent(f"""\
            Rewrite ONLY the single sentence provided.  Make it sound
            authentically human.  Keep the same meaning.  Output ONLY
            the rewritten sentence — no commentary.

            Known issues with this text:\n{diagnosis}
        """)

        fixed_count = 0
        for flat_idx, _sc in to_fix:
            para_idx, original_sent = all_sentences[flat_idx]
            # Retry up to 2 times on failure
            for attempt in range(2):
                try:
                    rewritten = self._ollama_chat(
                        model=model,
                        system_prompt=sys_prompt,
                        user_prompt=original_sent,
                        timeout=60,
                    )
                    if rewritten and 3 < len(rewritten.split()) < len(original_sent.split()) * 2:
                        all_sentences[flat_idx] = (para_idx, rewritten.strip())
                        fixed_count += 1
                        break
                except Exception as exc:
                    if attempt < 1:
                        time.sleep(1)
                        continue
                    log.warning("Sentence fix failed after retries: %s", exc)

        # Reconstruct text preserving paragraph structure
        rebuilt_paragraphs: List[List[str]] = [[] for _ in paragraphs]
        for para_idx, sent in all_sentences:
            rebuilt_paragraphs[para_idx].append(sent)

        result_paragraphs = [" ".join(sents) for sents in rebuilt_paragraphs]
        return "\n\n".join(result_paragraphs), fixed_count

    # ------------------------------------------------------------------
    # Full targeted rewrite (pass 2+)
    # ------------------------------------------------------------------

    def _targeted_rewrite(
        self, text: str, model: str, diagnosis: str,
        cfg: Optional[HumanizationConfig] = None,
    ) -> str:
        """Run a targeted full-text rewrite guided by detector diagnosis."""
        cfg = cfg or HumanizationConfig()

        sys_prompt = textwrap.dedent(f"""\
            You are an AI-text remediation specialist.  The user will give you:
            1. A DIAGNOSIS section listing the exact AI signals detected.
            2. The TEXT to fix.

            Your job: rewrite the text to specifically address EVERY flagged
            issue in the diagnosis while keeping ALL facts intact.

            Tone: {cfg.tone}.  Intensity: {cfg.intensity}.

            Guidelines:
            • If "low burstiness" → make sentence lengths dramatically uneven
            • If "transition overuse" → remove or replace every transition starter
            • If "low perplexity" → introduce surprises: unusual word choices,
              rhetorical questions, parenthetical asides, sentence fragments
            • If "uniformity" → break parallel structures, vary paragraph lengths
            • If "vocabulary" → swap jargon for plain language
            • If "passive voice" → convert to active voice

            Output ONLY the rewritten text.  No commentary, labels, or meta-text.
        """)

        user_msg = f"DIAGNOSIS:\n{diagnosis}\n\n---\n\nTEXT:\n{text}"
        try:
            result = self._ollama_chat(
                model=model,
                system_prompt=sys_prompt,
                user_prompt=user_msg,
                timeout=max(180, len(text.split()) * 3),
            )
            return result
        except Exception as exc:
            log.warning("Targeted rewrite failed: %s", exc)
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

    transition_freq = features.get("transition_word_ratio", 0)
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

    mean_sent_len = features.get("sentence_length_mean", 0)
    std_sent_len = features.get("sentence_length_std", 0)
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
    """Restructure the opening of a sentence for variety.

    Instead of prepending clichéd transition starters (which AI detectors
    flag), this applies genuine syntactic transformations:
    - Fronting an adverbial clause
    - Inverting subject/predicate order
    - Starting with a gerund phrase
    """
    words = sentence.split()
    if not words:
        return sentence
    first = words[0].lower()

    # Only rewrite sentences that start with common determiners/pronouns
    if first not in {"the", "this", "these", "it", "a", "an", "such"}:
        return sentence

    # Strategy 1: Move a prepositional/clause phrase to the front
    # Look for  ", which" / ", where" / ", because" to invert
    for marker in [", because ", ", since ", ", although "]:
        if marker in sentence:
            parts = sentence.split(marker, 1)
            if len(parts) == 2 and len(parts[1].split()) > 3:
                clause = parts[1].rstrip(".")
                main = parts[0]
                word = marker.strip(", ")
                return f"{word.capitalize()} {clause}, {main[0].lower()}{main[1:]}."

    # Strategy 2: For short sentences, just swap in a synonym for the determiner
    swaps = {
        "the": ["that particular", "one", "the very"],
        "this": ["that", "one such"],
        "these": ["such", "those"],
        "it": ["that"],
        "a": ["one", "a single"],
        "an": ["one"],
    }
    if first in swaps and random.random() < 0.5:
        replacement = random.choice(swaps[first])
        if words[0][0].isupper():
            replacement = replacement[0].upper() + replacement[1:]
        words[0] = replacement
        return " ".join(words)

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
    """Small random perturbations to sentence lengths for variability.

    Instead of adding clichéd "filler" transitions (which AI detectors flag),
    this creates length variety by occasionally:
    - Splitting a moderate sentence into two short ones
    - Appending a short follow-up fragment
    """
    result: List[str] = []
    for sent in sentences:
        words = sent.split()
        # Occasionally split a moderate sentence for burstiness
        if 12 < len(words) < 20 and random.random() < 0.12:
            mid = len(words) // 2
            # Find a natural break (after a comma)
            for offset in range(min(3, mid)):
                for pos in [mid + offset, mid - offset]:
                    if 0 < pos < len(words) and words[pos - 1].endswith(","):
                        a = " ".join(words[:pos]).rstrip(",") + "."
                        b = " ".join(words[pos:])
                        b = b[0].upper() + b[1:] if b else b
                        result.extend([a, b])
                        break
                else:
                    continue
                break
            else:
                result.append(sent)
        else:
            result.append(sent)
    return result
