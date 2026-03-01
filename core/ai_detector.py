"""
ai_detector.py
--------------
Hybrid AI Detection Engine for DocuGuard+.

Combines three independent signals into a final AI-probability score:

1. **Language-Model Perplexity** – How "predictable" the text is.  AI-generated
   text tends to have *low* perplexity compared to human writing.
2. **Stylometric Analysis** – Structural regularity, vocabulary uniformity, and
   sentence-level patterns typical of machine outputs.
3. **ML Classification Layer** – A lightweight scikit-learn classifier that fuses
   the above inputs into a calibrated probability.

All models are designed to run on **CPU** (no GPU required).
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from core.feature_extractor import extract_features, extract_feature_vector

# Suppress non-critical HF/tokenizer warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PerplexityResult:
    """Output of perplexity analysis."""
    perplexity: float = 0.0
    mean_token_loss: float = 0.0
    normalised_score: float = 0.0  # 0.0 = very human, 1.0 = very AI
    interpretation: str = ""


@dataclass
class StyleResult:
    """Output of stylometric analysis."""
    uniformity_score: float = 0.0       # 0 = varied (human), 1 = uniform (AI)
    vocabulary_score: float = 0.0       # higher → more AI-like
    naturalness_score: float = 0.0      # 0 = stiff (AI), 1 = natural (human)
    contributing_factors: List[str] = field(default_factory=list)


@dataclass
class AIDetectionResult:
    """Final combined result from the hybrid detector."""
    ai_probability: float = 0.0         # 0.0–1.0
    confidence: str = "Low"             # Low / Medium / High
    perplexity_result: PerplexityResult = field(default_factory=PerplexityResult)
    style_result: StyleResult = field(default_factory=StyleResult)
    features: Dict[str, Any] = field(default_factory=dict)
    explanations: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Perplexity Analyser (DistilGPT-2)
# ---------------------------------------------------------------------------

class PerplexityAnalyser:
    """Compute pseudo-perplexity with a causal LM (DistilGPT-2 by default).

    Lower perplexity → text is more predictable → higher AI likelihood.
    """

    _MODEL_NAME = "distilgpt2"

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    # Lazy loading to avoid slow import-time model download
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        cache_dir = os.environ.get("TRANSFORMERS_CACHE", None)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._MODEL_NAME, cache_dir=cache_dir
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._MODEL_NAME, cache_dir=cache_dir
        )
        self._model.eval()

    def analyse(
        self,
        text: str,
        window_size: int = 512,
        stride: int = 256,
    ) -> PerplexityResult:
        """Return perplexity metrics using a **sliding-window** approach.

        Instead of truncating the text to one 1024-token chunk (ignoring 95%
        of longer documents), we slide a *window_size*-token window across the
        full token sequence with *stride*-token steps and average the
        per-window losses.

        Parameters
        ----------
        text : str
            The input text.
        window_size : int
            Tokens per window (default 512 — fits comfortably in DistilGPT-2's
            1024-token context).
        stride : int
            Step size between windows (default 256 → 50 % overlap).
        """
        import torch

        self._ensure_loaded()

        # Tokenise the entire document (no truncation)
        encodings = self._tokenizer(
            text, return_tensors="pt", truncation=False
        )
        all_ids = encodings["input_ids"][0]  # shape: (total_tokens,)
        total_tokens = all_ids.shape[0]

        # Short-text fast-path: fits in a single window
        if total_tokens <= window_size:
            input_ids = all_ids.unsqueeze(0)
            with torch.no_grad():
                outputs = self._model(input_ids, labels=input_ids)
                loss = outputs.loss.item()
            ppl = math.exp(loss)
            normalised = self._normalise_ppl(ppl)
            return PerplexityResult(
                perplexity=round(ppl, 2),
                mean_token_loss=round(loss, 4),
                normalised_score=round(normalised, 4),
                interpretation=self._interpret(normalised),
            )

        # Sliding-window: collect per-window losses
        window_losses: List[float] = []
        for start in range(0, total_tokens, stride):
            end = min(start + window_size, total_tokens)
            if end - start < 32:
                break  # skip tiny tail windows
            chunk = all_ids[start:end].unsqueeze(0)
            with torch.no_grad():
                outputs = self._model(chunk, labels=chunk)
                window_losses.append(outputs.loss.item())

        if not window_losses:
            # Fallback (shouldn't happen)
            return PerplexityResult(interpretation="Insufficient text")

        mean_loss = sum(window_losses) / len(window_losses)
        ppl = math.exp(mean_loss)
        normalised = self._normalise_ppl(ppl)

        return PerplexityResult(
            perplexity=round(ppl, 2),
            mean_token_loss=round(mean_loss, 4),
            normalised_score=round(normalised, 4),
            interpretation=self._interpret(normalised),
        )

    # ---- helpers --
    @staticmethod
    def _normalise_ppl(ppl: float) -> float:
        """Sigmoid-based mapping: low PPL → score near 1 (AI)."""
        # Centre at 60, steepness 0.04
        return 1.0 / (1.0 + math.exp(0.04 * (ppl - 60)))

    @staticmethod
    def _interpret(score: float) -> str:
        if score >= 0.75:
            return "Very low perplexity — strong AI signal"
        if score >= 0.5:
            return "Moderately low perplexity — possible AI involvement"
        if score >= 0.3:
            return "Moderate perplexity — inconclusive"
        return "High perplexity — suggests human authorship"


# ---------------------------------------------------------------------------
# Stylometric Analyser
# ---------------------------------------------------------------------------

class StyleAnalyser:
    """Assess AI likelihood from the *full set* of stylometric features."""

    def analyse(self, features: Dict[str, Any]) -> StyleResult:
        factors: List[str] = []

        # ---- Sentence uniformity (low burstiness → AI-like) ----
        burstiness = features.get("burstiness", 0.5)
        if burstiness < 0.35:
            uniformity = 0.8
            factors.append(f"Low burstiness ({burstiness:.2f}) indicates uniform sentence lengths")
        elif burstiness < 0.55:
            uniformity = 0.5
        else:
            uniformity = 0.2
            factors.append(f"High burstiness ({burstiness:.2f}) suggests human-like variation")

        # ---- Vocabulary uniformity ----
        ttr = features.get("type_token_ratio", 0.5)
        hapax = features.get("hapax_ratio", 0.3)
        vocab_score = 0.0
        if ttr < 0.4:
            vocab_score += 0.4
            factors.append(f"Low TTR ({ttr:.3f}) — limited vocabulary range")
        if hapax < 0.3:
            vocab_score += 0.3
            factors.append(f"Low hapax ratio ({hapax:.3f}) — high word reuse")
        vocab_score = min(vocab_score, 1.0)

        # ---- Naturalness cues (expanded) ----
        naturalness = 0.5
        contraction = features.get("contraction_ratio", 0.0)
        starter_div = features.get("sentence_starter_diversity", 0.5)
        transition = features.get("transition_word_ratio", 0.0)

        if contraction > 0.005:
            naturalness += 0.12
            factors.append("Contractions present — human cue")
        if starter_div > 0.7:
            naturalness += 0.10
            factors.append(f"Higher sentence-starter diversity ({starter_div:.2f})")
        if transition > 0.02:
            naturalness -= 0.12
            factors.append(f"Heavy transition-word use ({transition:.3f}) — AI pattern")

        # Passive voice — AI tends toward more passive constructions
        passive = features.get("passive_voice_ratio", 0.0)
        if passive > 0.25:
            naturalness -= 0.08
            factors.append(f"High passive voice ratio ({passive:.2f}) — common in AI text")
        elif passive > 0.0:
            naturalness += 0.05

        # Flesch reading ease — AI usually targets mid-range (40-60)
        flesch = features.get("flesch_reading_ease", 50.0)
        if 35 < flesch < 65:
            naturalness -= 0.06
            factors.append(f"Flesch score ({flesch:.1f}) in narrow AI-typical band")
        elif flesch > 70 or flesch < 25:
            naturalness += 0.06
            factors.append(f"Flesch score ({flesch:.1f}) outside typical AI range")

        # Repeated n-gram phrases — AI avoids them; humans repeat
        repeated = features.get("repeated_phrase_ratio", 0.0)
        if repeated > 0.02:
            naturalness += 0.06
            factors.append(f"Repeated phrases ({repeated:.3f}) — natural repetition")
        elif repeated < 0.005:
            naturalness -= 0.04

        # Average paragraph length — AI makes very even paragraphs
        para_mean = features.get("paragraph_length_mean", 0)
        para_std = features.get("paragraph_length_std", 0)
        if para_mean > 0 and para_std / (para_mean + 1e-9) < 0.2:
            naturalness -= 0.06
            factors.append("Very uniform paragraph lengths — AI pattern")

        # Conjunction ratio — AI overuses coordinating conjunctions
        conj = features.get("conjunction_ratio", 0.0)
        if conj > 0.04:
            naturalness -= 0.05
            factors.append(f"High conjunction ratio ({conj:.3f})")

        naturalness = max(0.0, min(naturalness, 1.0))

        return StyleResult(
            uniformity_score=round(uniformity, 4),
            vocabulary_score=round(vocab_score, 4),
            naturalness_score=round(naturalness, 4),
            contributing_factors=factors,
        )


# ---------------------------------------------------------------------------
# ML Classification Layer
# ---------------------------------------------------------------------------

_MODEL_DIR = os.environ.get("MODEL_DIR", "models")
_CLASSIFIER_PATH = os.path.join(_MODEL_DIR, "ai_classifier.joblib")


class MLClassifier:
    """Light-weight classifier that fuses perplexity + style signals.

    If a trained model exists on disk, it will be loaded.  Otherwise the class
    falls back to a **weighted-heuristic** combiner so the system is usable
    out of the box without historical training data.
    """

    def __init__(self) -> None:
        self._model = None
        self._trained = False
        if os.path.isfile(_CLASSIFIER_PATH):
            try:
                self._model = joblib.load(_CLASSIFIER_PATH)
                self._trained = True
            except Exception:
                pass

    @property
    def is_trained(self) -> bool:
        return self._trained

    def predict(
        self,
        perplexity_score: float,
        style: StyleResult,
        feature_vector: List[float],
    ) -> float:
        """Return an AI probability in [0, 1]."""
        if self._trained and self._model is not None:
            # Feature order: perplexity_norm, uniformity, vocab, naturalness, ...feature_vec
            x = np.array(
                [perplexity_score, style.uniformity_score,
                 style.vocabulary_score, 1.0 - style.naturalness_score]
                + feature_vector
            ).reshape(1, -1)
            proba = self._model.predict_proba(x)[0]
            # Assume class 1 = AI
            return float(proba[1]) if len(proba) > 1 else float(proba[0])

        # ---- Heuristic fallback ----
        return self._heuristic(perplexity_score, style)

    @staticmethod
    def _heuristic(perplexity_score: float, style: StyleResult) -> float:
        """Weighted combination when no trained model is available."""
        w_ppl = 0.45
        w_uniform = 0.20
        w_vocab = 0.15
        w_natural = 0.20

        score = (
            w_ppl * perplexity_score
            + w_uniform * style.uniformity_score
            + w_vocab * style.vocabulary_score
            + w_natural * (1.0 - style.naturalness_score)
        )
        return max(0.0, min(1.0, round(score, 4)))

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Train (or retrain) the classifier and persist to disk.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        y : array of shape (n_samples,) with labels 0=human, 1=AI
        """
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X, y)
        os.makedirs(_MODEL_DIR, exist_ok=True)
        joblib.dump(clf, _CLASSIFIER_PATH)
        self._model = clf
        self._trained = True


# ---------------------------------------------------------------------------
# Facade: AIDetector
# ---------------------------------------------------------------------------

class AIDetector:
    """Hybrid AI detection engine combining perplexity, style, and ML."""

    def __init__(self) -> None:
        self._perplexity = PerplexityAnalyser()
        self._style = StyleAnalyser()
        self._classifier = MLClassifier()

    def detect(self, text: str) -> AIDetectionResult:
        """Run the full detection pipeline on *text*.

        Returns
        -------
        AIDetectionResult
            Contains the overall probability, confidence level, sub-results,
            and human-readable explanations.
        """
        # 1. Feature extraction
        features = extract_features(text)
        feature_vec = extract_feature_vector(text)

        # 2. Perplexity analysis
        ppl_result = self._perplexity.analyse(text)

        # 3. Stylometric analysis
        style_result = self._style.analyse(features)

        # 4. ML classification (or heuristic fallback)
        ai_prob = self._classifier.predict(
            ppl_result.normalised_score, style_result, feature_vec
        )

        # 5. Confidence estimation
        confidence = self._confidence_level(
            ppl_result.normalised_score,
            style_result.uniformity_score,
            ai_prob,
        )

        # 6. Build explanations
        explanations = self._build_explanations(
            ai_prob, ppl_result, style_result, features
        )

        return AIDetectionResult(
            ai_probability=round(ai_prob, 4),
            confidence=confidence,
            perplexity_result=ppl_result,
            style_result=style_result,
            features=features,
            explanations=explanations,
        )

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _confidence_level(ppl_score: float, uniformity: float, prob: float) -> str:
        """Derive a qualitative confidence label."""
        agreement = 1.0 - abs(ppl_score - uniformity)
        if agreement > 0.7 and (prob > 0.75 or prob < 0.25):
            return "High"
        if agreement > 0.4 or prob > 0.65 or prob < 0.35:
            return "Medium"
        return "Low"

    @staticmethod
    def _build_explanations(
        prob: float,
        ppl: PerplexityResult,
        style: StyleResult,
        features: Dict[str, Any],
    ) -> List[str]:
        explanations: List[str] = []

        # Main verdict
        if prob >= 0.75:
            explanations.append(
                f"Overall AI probability is high ({prob:.0%}). "
                "Multiple signals point to machine-generated content."
            )
        elif prob >= 0.45:
            explanations.append(
                f"Overall AI probability is moderate ({prob:.0%}). "
                "The text shows a mix of human and AI characteristics."
            )
        else:
            explanations.append(
                f"Overall AI probability is low ({prob:.0%}). "
                "Indicators lean towards human authorship."
            )

        # Perplexity insight
        explanations.append(f"Perplexity: {ppl.perplexity:.1f} — {ppl.interpretation}")

        # Feature-specific insights
        burstiness = features.get("burstiness", 0)
        if burstiness > 0.55:
            explanations.append(
                f"High burstiness ({burstiness:.2f}) suggests natural sentence-length "
                "variation typical of human writing."
            )
        elif burstiness < 0.35:
            explanations.append(
                f"Low burstiness ({burstiness:.2f}) indicates uniform sentence lengths "
                "often seen in AI outputs."
            )

        # Stylometric factors
        explanations.extend(style.contributing_factors)

        return explanations
