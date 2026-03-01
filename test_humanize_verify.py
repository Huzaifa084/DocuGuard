"""Verify the full multi-pass LLM humanization pipeline."""
import sys, os, time
sys.path.insert(0, ".")
os.environ["OLLAMA_URL"] = "http://localhost:11434"

from core.humanizer import Humanizer
from core.ai_detector import AIDetector

h = Humanizer()
det = AIDetector()

# Clearly AI-style text (repetitive transitions, uniform sentences, formal tone)
AI_TEXT = """Furthermore, artificial intelligence has fundamentally transformed the landscape of modern education. Moreover, the integration of machine learning algorithms has enabled unprecedented levels of personalization in student learning pathways. Additionally, the development of natural language processing capabilities has significantly advanced automated essay scoring systems. Consequently, these systems analyze textual features including vocabulary diversity, syntactic complexity, and coherence patterns. Furthermore, the adoption of intelligent tutoring systems has demonstrated measurable improvements in student outcomes. Moreover, the widespread deployment of AI in education raises important ethical considerations. Additionally, questions regarding data privacy and algorithmic bias must be carefully addressed. Consequently, the reliability of AI-generated content detection tools remains an area of active research."""

print("=" * 70)
print("ORIGINAL TEXT")
print("=" * 70)
print(AI_TEXT)
print(f"\nWord count: {len(AI_TEXT.split())}")

# --- AI Detection BEFORE ---
ai_before = det.detect(AI_TEXT)
print(f"\nAI Probability (BEFORE): {ai_before.ai_probability:.2%}")
print(f"Confidence: {ai_before.confidence}")
print(f"Perplexity: {ai_before.perplexity_result.perplexity:.2f}")
print(f"Burstiness: {ai_before.features.get('burstiness', 'N/A')}")

# --- Rule-based humanization ---
print("\n" + "=" * 70)
print("RULE-BASED HUMANIZATION")
print("=" * 70)
h_rule = h.humanize(AI_TEXT, strategy="rule-based")
print(h_rule.humanized_text)
print(f"\nChanges: {h_rule.changes_summary}")

ai_after_rule = det.detect(h_rule.humanized_text)
print(f"AI Probability (AFTER rule-based): {ai_after_rule.ai_probability:.2%}")
print(f"Delta: {ai_before.ai_probability - ai_after_rule.ai_probability:+.2%}")

# --- LLM humanization (multi-pass pipeline) ---
print("\n" + "=" * 70)
print("LLM HUMANIZATION — MULTI-PASS PIPELINE (MISTRAL)")
print("=" * 70)

def progress(stage, pct):
    print(f"  [{pct:5.0%}] {stage}")

t0 = time.time()
h_llm = h.humanize(AI_TEXT, strategy="llm", model="mistral", progress_cb=progress)
elapsed = time.time() - t0

print(f"\n--- Pipeline completed in {elapsed:.1f}s ---")
print(f"\nHumanized text:\n{h_llm.humanized_text}")
print(f"\nStrategy: {h_llm.strategy}")
print(f"Pipeline stages:")
for ch in h_llm.changes_summary:
    print(f"  • {ch}")

ai_after_llm = det.detect(h_llm.humanized_text)
print(f"\nAI Probability (AFTER LLM pipeline): {ai_after_llm.ai_probability:.2%}")
print(f"Burstiness (AFTER): {ai_after_llm.features.get('burstiness', 'N/A')}")
print(f"Delta: {ai_before.ai_probability - ai_after_llm.ai_probability:+.2%}")

# --- Summary ---
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Original AI prob:     {ai_before.ai_probability:.2%}")
print(f"  After rule-based:     {ai_after_rule.ai_probability:.2%}  ({ai_before.ai_probability - ai_after_rule.ai_probability:+.2%})")
print(f"  After LLM pipeline:   {ai_after_llm.ai_probability:.2%}  ({ai_before.ai_probability - ai_after_llm.ai_probability:+.2%})")
print(f"\n  LLM pipeline time:    {elapsed:.1f}s")
print(f"  Text actually changed: {'YES' if AI_TEXT.strip() != h_llm.humanized_text.strip() else 'NO (PROBLEM!)'}")
