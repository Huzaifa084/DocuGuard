"""
fingerprint.py
--------------
Writing Fingerprint Comparison module for DocuGuard+.

Allows users to upload their own past work to create a **Personal Style
Profile**.  New documents are then compared against this profile to produce
a "Style Similarity Score", helping maintain a consistent academic voice.

Profiles are stored as JSON files under the ``fingerprints/`` directory.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from core.feature_extractor import extract_features, extract_feature_vector

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FINGERPRINT_DIR = os.environ.get("FINGERPRINT_DIR", "fingerprints")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StyleProfile:
    """Persisted writing-style profile built from one or more documents."""
    profile_name: str = "default"
    num_documents: int = 0
    mean_vector: List[float] = field(default_factory=list)
    std_vector: List[float] = field(default_factory=list)
    mean_features: Dict[str, float] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)


@dataclass
class FingerprintResult:
    """Comparison result between a document and a style profile."""
    similarity_score: float = 0.0       # 0–1 (cosine similarity)
    deviation_summary: Dict[str, float] = field(default_factory=dict)
    interpretation: str = ""
    profile_name: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FingerprintEngine:
    """Build, persist, and compare writing-style fingerprints."""

    def __init__(self, profile_dir: str = FINGERPRINT_DIR) -> None:
        self.profile_dir = profile_dir
        os.makedirs(self.profile_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Profile creation
    # ------------------------------------------------------------------

    def create_profile(
        self,
        texts: List[str],
        profile_name: str = "default",
    ) -> StyleProfile:
        """Build a style profile from a list of document texts.

        Each text is feature-extracted, and the profile stores the per-feature
        mean and standard deviation across all documents.
        """
        if not texts:
            raise ValueError("At least one document text is required.")

        vectors: List[List[float]] = []
        all_features: List[Dict[str, Any]] = []

        for t in texts:
            vec = extract_feature_vector(t)
            feats = extract_features(t)
            vectors.append(vec)
            all_features.append(feats)

        arr = np.array(vectors, dtype=np.float64)
        mean_vec = arr.mean(axis=0).tolist()
        std_vec = arr.std(axis=0).tolist() if arr.shape[0] > 1 else [0.0] * arr.shape[1]

        # Per-feature mean (human-readable)
        numeric_keys = sorted(
            k for k in all_features[0] if isinstance(all_features[0][k], (int, float))
        )
        mean_feats: Dict[str, float] = {}
        for k in numeric_keys:
            vals = [f[k] for f in all_features if k in f]
            mean_feats[k] = round(float(np.mean(vals)), 4) if vals else 0.0

        profile = StyleProfile(
            profile_name=profile_name,
            num_documents=len(texts),
            mean_vector=mean_vec,
            std_vector=std_vec,
            mean_features=mean_feats,
            feature_names=numeric_keys,
        )

        self._save_profile(profile)
        return profile

    def update_profile(
        self,
        new_texts: List[str],
        profile_name: str = "default",
    ) -> StyleProfile:
        """Incrementally update an existing profile with new documents.

        Uses running-mean / running-std formulas so previously uploaded
        documents don't need to be re-processed.  If no profile exists yet,
        falls back to ``create_profile``.
        """
        existing = self.load_profile(profile_name)
        if existing is None:
            return self.create_profile(new_texts, profile_name)

        # Extract features from new texts
        new_vectors: List[List[float]] = []
        new_features: List[Dict[str, Any]] = []
        for t in new_texts:
            new_vectors.append(extract_feature_vector(t))
            new_features.append(extract_features(t))

        new_arr = np.array(new_vectors, dtype=np.float64)
        new_mean = new_arr.mean(axis=0)
        old_mean = np.array(existing.mean_vector, dtype=np.float64)
        old_std = np.array(existing.std_vector, dtype=np.float64)

        n_old = existing.num_documents
        n_new = len(new_texts)
        n_total = n_old + n_new

        # Combined mean
        combined_mean = (old_mean * n_old + new_mean * n_new) / n_total

        # Combined std (via parallel variance formula)
        if n_new > 1:
            new_std = new_arr.std(axis=0)
        else:
            new_std = np.zeros_like(new_mean)
        old_var = old_std ** 2
        new_var = new_std ** 2
        combined_var = (
            (n_old * (old_var + (old_mean - combined_mean) ** 2)
             + n_new * (new_var + (new_mean - combined_mean) ** 2))
            / n_total
        )
        combined_std = np.sqrt(combined_var)

        # Per-feature human-readable means
        numeric_keys = sorted(existing.mean_features.keys())
        mean_feats: Dict[str, float] = {}
        for k in numeric_keys:
            old_val = existing.mean_features.get(k, 0.0)
            new_vals = [f[k] for f in new_features if k in f]
            if new_vals:
                new_val = float(np.mean(new_vals))
                mean_feats[k] = round(
                    (old_val * n_old + new_val * n_new) / n_total, 4
                )
            else:
                mean_feats[k] = old_val

        profile = StyleProfile(
            profile_name=profile_name,
            num_documents=n_total,
            mean_vector=combined_mean.tolist(),
            std_vector=combined_std.tolist(),
            mean_features=mean_feats,
            feature_names=numeric_keys,
        )
        self._save_profile(profile)
        return profile

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        text: str,
        profile_name: str = "default",
    ) -> FingerprintResult:
        """Compare *text* against a stored style profile.

        Uses **z-score normalisation** before cosine similarity so that
        features with large absolute values (e.g. ``sentence_count``,
        Yule's K) don't dominate the comparison.

        Returns
        -------
        FingerprintResult
            Contains similarity score, per-feature deviations, and an
            interpretation string.
        """
        profile = self.load_profile(profile_name)
        if profile is None:
            return FingerprintResult(
                interpretation=f"No profile '{profile_name}' found. "
                               "Upload your own past work to create one.",
                profile_name=profile_name,
            )

        doc_vec = np.array(extract_feature_vector(text), dtype=np.float64)
        profile_mean = np.array(profile.mean_vector, dtype=np.float64)
        profile_std = np.array(profile.std_vector, dtype=np.float64)

        # Ensure vectors have compatible shapes
        min_len = min(len(doc_vec), len(profile_mean))
        doc_vec = doc_vec[:min_len]
        profile_mean = profile_mean[:min_len]
        profile_std = profile_std[:min_len] if len(profile_std) >= min_len else np.zeros(min_len)

        # Z-score normalisation: centres each feature around the profile mean
        # and scales by the profile std.  This prevents high-magnitude features
        # like sentence_count or lexical_diversity from dominating cosine sim.
        safe_std = np.where(profile_std > 1e-9, profile_std, 1.0)
        doc_z = (doc_vec - profile_mean) / safe_std
        profile_z = np.zeros_like(doc_z)  # profile mean maps to the origin

        # Cosine similarity on normalised vectors
        # When all deviations are zero doc_z is the zero vector → sim = 1.0
        doc_norm = np.linalg.norm(doc_z)
        if doc_norm < 1e-9:
            sim = 1.0  # Identical to profile
        else:
            sim = float(cosine_similarity(
                doc_z.reshape(1, -1), profile_z.reshape(1, -1)
            )[0, 0])
            # cosine_similarity returns -1..1; map to 0..1 for display
            sim = (sim + 1.0) / 2.0

        sim = max(0.0, min(1.0, sim))

        # Per-feature deviation (human-readable)
        doc_feats = extract_features(text)
        deviations: Dict[str, float] = {}
        for k, prof_val in profile.mean_features.items():
            if k in doc_feats and isinstance(doc_feats[k], (int, float)):
                diff = float(doc_feats[k]) - prof_val
                deviations[k] = round(diff, 4)

        interpretation = self._interpret(sim, deviations)

        return FingerprintResult(
            similarity_score=round(sim, 4),
            deviation_summary=deviations,
            interpretation=interpretation,
            profile_name=profile_name,
        )

    # ------------------------------------------------------------------
    # Profile persistence
    # ------------------------------------------------------------------

    def load_profile(self, profile_name: str) -> Optional[StyleProfile]:
        path = self._profile_path(profile_name)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return StyleProfile(**data)

    def list_profiles(self) -> List[str]:
        """Return names of all saved profiles."""
        names: List[str] = []
        if not os.path.isdir(self.profile_dir):
            return names
        for fname in os.listdir(self.profile_dir):
            if fname.endswith(".json"):
                names.append(fname.removesuffix(".json"))
        return sorted(names)

    def delete_profile(self, profile_name: str) -> bool:
        path = self._profile_path(profile_name)
        if os.path.isfile(path):
            os.remove(path)
            return True
        return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _profile_path(self, name: str) -> str:
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        return os.path.join(self.profile_dir, f"{safe_name}.json")

    def _save_profile(self, profile: StyleProfile) -> None:
        path = self._profile_path(profile.profile_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(profile), f, indent=2)

    @staticmethod
    def _interpret(sim: float, deviations: Dict[str, float]) -> str:
        if sim >= 0.92:
            base = "Very high similarity — the writing style closely matches your profile."
        elif sim >= 0.80:
            base = "Good similarity — the text is broadly consistent with your style."
        elif sim >= 0.60:
            base = "Moderate similarity — some stylistic differences from your typical writing."
        else:
            base = "Low similarity — this text diverges significantly from your writing profile."

        # Highlight the biggest deviations
        if deviations:
            sorted_devs = sorted(deviations.items(), key=lambda x: abs(x[1]), reverse=True)
            top = sorted_devs[:3]
            details = "; ".join(
                f"{k}: {'+'if v>0 else ''}{v:.3f}" for k, v in top
            )
            base += f" Top deviations: {details}."

        return base
