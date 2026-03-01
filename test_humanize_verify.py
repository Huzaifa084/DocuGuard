"""Verify humanization is actually changing text meaningfully."""
import sys, os
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

# --- LLM humanization (Mistral) ---
print("\n" + "=" * 70)
print("LLM HUMANIZATION (MISTRAL)")
print("=" * 70)
h_llm = h.humanize(AI_TEXT, strategy="llm", model="mistral")
print(h_llm.humanized_text)
print(f"\nStrategy used: {h_llm.strategy}")
print(f"Changes: {h_llm.changes_summary}")

ai_after_llm = det.detect(h_llm.humanized_text)
print(f"AI Probability (AFTER Mistral): {ai_after_llm.ai_probability:.2%}")
print(f"Delta: {ai_before.ai_probability - ai_after_llm.ai_probability:+.2%}")

# --- Side-by-side sentence comparison ---
print("\n" + "=" * 70)
print("SENTENCE-BY-SENTENCE COMPARISON (Original vs Mistral)")
print("=" * 70)
from utils.text_utils import split_sentences
orig_sents = split_sentences(AI_TEXT)
llm_sents = split_sentences(h_llm.humanized_text)

for i, s in enumerate(orig_sents):
    print(f"\n[Original {i+1}]: {s.strip()}")
    if i < len(llm_sents):
        print(f"[Mistral  {i+1}]: {llm_sents[i].strip()}")
    else:
        print(f"[Mistral  {i+1}]: (no corresponding sentence)")

if len(llm_sents) > len(orig_sents):
    for i in range(len(orig_sents), len(llm_sents)):
        print(f"\n[Mistral  {i+1}] (extra): {llm_sents[i].strip()}")

# --- Summary ---
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Original AI prob:     {ai_before.ai_probability:.2%}")
print(f"  After rule-based:     {ai_after_rule.ai_probability:.2%}  ({ai_before.ai_probability - ai_after_rule.ai_probability:+.2%})")
print(f"  After Mistral LLM:    {ai_after_llm.ai_probability:.2%}  ({ai_before.ai_probability - ai_after_llm.ai_probability:+.2%})")
print(f"\n  Rule-based changed:   {len(h_rule.changes_summary)} transforms")
print(f"  LLM strategy used:    {h_llm.strategy}")
same_text = AI_TEXT.strip() == h_llm.humanized_text.strip()
print(f"  Text actually changed: {'NO (PROBLEM!)' if same_text else 'YES'}")
