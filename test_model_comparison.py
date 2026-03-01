"""Head-to-head comparison: Mistral vs Llama 3 for humanization quality."""
import sys, os, time
sys.path.insert(0, ".")
os.environ["OLLAMA_URL"] = "http://localhost:11434"

from core.humanizer import Humanizer
from core.ai_detector import AIDetector

h = Humanizer()
det = AIDetector()

AI_TEXT = """Furthermore, artificial intelligence has fundamentally transformed the landscape of modern education. Moreover, the integration of machine learning algorithms has enabled unprecedented levels of personalization in student learning pathways. Additionally, the development of natural language processing capabilities has significantly advanced automated essay scoring systems. Consequently, these systems analyze textual features including vocabulary diversity, syntactic complexity, and coherence patterns. Furthermore, the adoption of intelligent tutoring systems has demonstrated measurable improvements in student outcomes. Moreover, the widespread deployment of AI in education raises important ethical considerations. Additionally, questions regarding data privacy and algorithmic bias must be carefully addressed. Consequently, the reliability of AI-generated content detection tools remains an area of active research."""

# --- Baseline ---
ai_before = det.detect(AI_TEXT)
print(f"ORIGINAL  →  AI prob: {ai_before.ai_probability:.2%}  |  "
      f"Burstiness: {ai_before.features.get('burstiness',0):.3f}  |  "
      f"Perplexity: {ai_before.perplexity_result.perplexity:.1f}")
print("=" * 80)

results = {}

for model in ["mistral", "llama3"]:
    print(f"\n{'─' * 80}")
    print(f"  MODEL: {model.upper()}")
    print(f"{'─' * 80}")

    t0 = time.time()
    h_result = h.humanize(
        AI_TEXT, strategy="llm", model=model,
        progress_cb=lambda stage, pct: print(f"  [{pct:5.0%}] {stage}"),
    )
    elapsed = time.time() - t0

    ai_after = det.detect(h_result.humanized_text)

    results[model] = {
        "ai_prob": ai_after.ai_probability,
        "burstiness": ai_after.features.get("burstiness", 0),
        "perplexity": ai_after.perplexity_result.perplexity,
        "ttr": ai_after.features.get("type_token_ratio", 0),
        "starter_div": ai_after.features.get("sentence_starter_diversity", 0),
        "transition_freq": ai_after.features.get("transition_word_freq", 0),
        "elapsed": elapsed,
        "text": h_result.humanized_text,
        "stages": h_result.changes_summary,
        "word_count": len(h_result.humanized_text.split()),
    }

    r = results[model]
    print(f"\n  Output ({r['word_count']} words):")
    print(f"  {h_result.humanized_text}\n")
    print(f"  Pipeline stages:")
    for s in h_result.changes_summary:
        print(f"    • {s}")
    print(f"\n  AI Probability:       {r['ai_prob']:.2%}")
    print(f"  Burstiness:           {r['burstiness']:.3f}")
    print(f"  Perplexity:           {r['perplexity']:.1f}")
    print(f"  Type-Token Ratio:     {r['ttr']:.3f}")
    print(f"  Starter Diversity:    {r['starter_div']:.3f}")
    print(f"  Transition Freq:      {r['transition_freq']:.4f}")
    print(f"  Time:                 {r['elapsed']:.1f}s")

# --- Final comparison table ---
print("\n" + "=" * 80)
print("  FINAL COMPARISON")
print("=" * 80)

header = f"{'Metric':<25} {'Original':>12} {'Mistral':>12} {'Llama 3':>12}  {'Winner':>10}"
print(header)
print("─" * len(header))

def winner(metric, lower_is_better=True):
    m = results["mistral"][metric]
    l = results["llama3"][metric]
    if lower_is_better:
        return "mistral" if m < l else "llama3" if l < m else "tie"
    return "mistral" if m > l else "llama3" if l > m else "tie"

orig_b = ai_before.features.get("burstiness", 0)
orig_ppl = ai_before.perplexity_result.perplexity

rows = [
    ("AI Probability ↓",   f"{ai_before.ai_probability:.2%}", True),
    ("Burstiness ↑",       f"{orig_b:.3f}", False),
    ("Perplexity ↑",       f"{orig_ppl:.1f}", False),
    ("Type-Token Ratio ↑", f"{ai_before.features.get('type_token_ratio',0):.3f}", False),
    ("Starter Diversity ↑", f"{ai_before.features.get('sentence_starter_diversity',0):.3f}", False),
    ("Transition Freq ↓",  f"{ai_before.features.get('transition_word_freq',0):.4f}", True),
    ("Time (s) ↓",         "—", True),
]

metric_keys = ["ai_prob", "burstiness", "perplexity", "ttr", "starter_div", "transition_freq", "elapsed"]

for (label, orig_val, lower_best), key in zip(rows, metric_keys):
    m = results["mistral"][key]
    l = results["llama3"][key]
    w = winner(key, lower_is_better=lower_best)
    fmt = ".2%" if "prob" in key.lower() else ".3f" if key in ("burstiness","ttr","starter_div") else ".4f" if "freq" in key else ".1f"
    print(f"{label:<25} {orig_val:>12} {format(m, fmt):>12} {format(l, fmt):>12}  {'← ' + w.upper():>10}")

print()
m_prob = results["mistral"]["ai_prob"]
l_prob = results["llama3"]["ai_prob"]
best = "MISTRAL" if m_prob < l_prob else "LLAMA 3" if l_prob < m_prob else "TIE"
print(f"  🏆  WINNER (by AI probability): {best}")
print(f"      Mistral: {ai_before.ai_probability:.2%} → {m_prob:.2%}  (Δ {ai_before.ai_probability - m_prob:+.2%})")
print(f"      Llama 3: {ai_before.ai_probability:.2%} → {l_prob:.2%}  (Δ {ai_before.ai_probability - l_prob:+.2%})")
