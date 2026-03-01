"""Test LLM humanization end-to-end with Mistral via Ollama."""
import sys, os
sys.path.insert(0, ".")
os.environ["OLLAMA_URL"] = "http://localhost:11434"

from core.humanizer import Humanizer
h = Humanizer()

test_text = (
    "Artificial intelligence has fundamentally transformed the landscape of modern technology. "
    "Furthermore, the integration of machine learning algorithms has enabled unprecedented "
    "levels of automation in various industries. Moreover, the development of neural networks "
    "has significantly advanced the field of natural language processing. Additionally, the "
    "widespread adoption of AI systems has raised important ethical considerations that "
    "society must address."
)

print("=== Testing LLM Humanization (Mistral) ===")
print(f"Input length: {len(test_text.split())} words")
print()

result = h.humanize(test_text, strategy="llm", model="mistral")
print(f"Strategy: {result.strategy}")
print(f"Changes: {result.changes_summary}")
print()
print("--- Original ---")
print(test_text[:200])
print()
print("--- Humanized ---")
print(result.humanized_text[:500])
print()

# Now test re-evaluation loop
from core.ai_detector import AIDetector
det = AIDetector()

print("=== Re-evaluation Loop ===")
ai_before = det.detect(test_text)
print(f"AI Prob (Before): {ai_before.ai_probability:.2%}")

ai_after = det.detect(result.humanized_text)
print(f"AI Prob (After):  {ai_after.ai_probability:.2%}")

delta = ai_before.ai_probability - ai_after.ai_probability
print(f"Improvement:      {delta:+.2%}")
