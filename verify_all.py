"""Quick verification of all DocuGuard+ components."""
import sys
sys.path.insert(0, ".")

# Test 1: Document Processor
from core.document_processor import process_document
d = process_document(b"Hello world. This is test.", "test.txt")
print(f"[OK] Document Processor: {d.metadata.word_count} words")

# Test 2: Feature Extractor
from core.feature_extractor import extract_features
f = extract_features("The quick brown fox jumps over the lazy dog repeatedly.")
print(f"[OK] Feature Extractor: {len(f)} features extracted")

# Test 3: AI Detector
from core.ai_detector import AIDetector
det = AIDetector()
r = det.detect("Artificial intelligence has transformed many industries.")
print(f"[OK] AI Detector: prob={r.ai_probability:.2%}, conf={r.confidence}")

# Test 4: Plagiarism Detector
from core.plagiarism_detector import PlagiarismDetector
pd = PlagiarismDetector()
pr = pd.check("Machine learning is a subset of artificial intelligence.")
print(f"[OK] Plagiarism Detector: lexical={pr.lexical_score:.2%}, semantic={pr.semantic_score:.2%}")

# Test 5: Humanizer (rule-based)
from core.humanizer import Humanizer
h = Humanizer()
hr = h.humanize(
    "Furthermore, the system demonstrates significant improvements. Moreover, the results are consistent.",
    strategy="rule-based",
)
print(f"[OK] Humanizer (rule): {len(hr.changes_summary)} changes")

# Test 6: Fingerprint
from core.fingerprint import FingerprintEngine
fp = FingerprintEngine()
p = fp.create_profile("test_verify", ["This is sample text for fingerprinting."])
print(f"[OK] Fingerprint: {len(p.mean_vector)} dims")
fp.delete_profile("test_verify")

# Test 7: Report Generator
from core.report_generator import generate_report
rep = generate_report(d.metadata, r, pr)
print(f"[OK] Report Generator: {len(rep)} report keys")

# Test 8: DB layer
from db.chroma_client import get_client
c = get_client()
print(f"[OK] ChromaDB: connected")

# Test 9: Ollama connectivity
import requests
try:
    resp = requests.get("http://localhost:11434/api/tags", timeout=3)
    models = [m["name"] for m in resp.json().get("models", [])]
    status = ", ".join(models) if models else "downloading..."
    print(f"[OK] Ollama: connected, models={status}")
except Exception as e:
    print(f"[FAIL] Ollama: not reachable ({e})")

# Test 10: Humanizer LLM connectivity (dry check)
try:
    resp = requests.get("http://localhost:11434/", timeout=3)
    print(f"[OK] Ollama API: HTTP {resp.status_code}")
except Exception as e:
    print(f"[FAIL] Ollama API: {e}")

print("\n=== All core modules verified ===")
