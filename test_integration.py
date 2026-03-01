"""Quick integration test for all DocuGuard+ core modules."""

import sys
import traceback


def test_text_utils():
    from utils.text_utils import clean_text, split_sentences, tokenize_words, flesch_reading_ease
    text = "The quick brown fox jumps.  This is a test!  Short."
    assert clean_text("  hello   world  ") == "hello world"
    assert len(split_sentences(text)) >= 2
    assert len(tokenize_words(text)) > 0
    fre = flesch_reading_ease(text)
    assert isinstance(fre, float)
    print(f"  [OK] text_utils  (FRE={fre:.1f})")


def test_document_processor():
    from core.document_processor import process_document
    sample = b"This is a sample document for testing. It has multiple sentences. Here is a third one."
    result = process_document(sample, "test.txt")
    assert result.metadata.word_count > 0
    assert result.metadata.sentence_count >= 3
    assert len(result.sentences) >= 3
    print(f"  [OK] document_processor  (words={result.metadata.word_count}, sents={result.metadata.sentence_count})")


def test_feature_extractor():
    from core.feature_extractor import extract_features, extract_feature_vector, feature_names
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "This is a test sentence with some variation! "
        "Short sentences help. But longer ones with more complexity "
        "also play an important role in demonstrating burstiness."
    )
    feats = extract_features(text)
    vec = extract_feature_vector(text)
    names = feature_names()
    assert "burstiness" in feats
    assert "type_token_ratio" in feats
    assert len(vec) == len(names)
    print(f"  [OK] feature_extractor  (features={len(feats)}, vector_len={len(vec)})")
    print(f"       burstiness={feats['burstiness']}, ttr={feats['type_token_ratio']}")


def test_ai_detector():
    from core.ai_detector import AIDetector
    det = AIDetector()
    text = (
        "Machine learning is a subset of artificial intelligence. "
        "It involves training algorithms on data. The algorithms learn "
        "patterns from the data. These patterns help make predictions. "
        "Deep learning uses neural networks with many layers."
    )
    result = det.detect(text)
    assert 0.0 <= result.ai_probability <= 1.0
    assert result.confidence in ("Low", "Medium", "High")
    assert len(result.explanations) > 0
    print(f"  [OK] ai_detector  (prob={result.ai_probability:.2%}, "
          f"conf={result.confidence}, ppl={result.perplexity_result.perplexity:.1f})")


def test_plagiarism_detector():
    from core.plagiarism_detector import PlagiarismDetector
    det = PlagiarismDetector()
    result = det.check("This is a test document for plagiarism checking.")
    # Corpus is empty, so score should be low / empty
    assert result.overall_score == 0.0 or result.disclaimer
    print(f"  [OK] plagiarism_detector  (score={result.overall_score:.2%}, "
          f"disclaimer present={bool(result.disclaimer)})")


def test_humanizer():
    from core.humanizer import Humanizer
    h = Humanizer()
    text = (
        "However, this is a very formal sentence. "
        "Furthermore, there is no variation. "
        "Additionally, every sentence starts the same way. "
        "Moreover, this is not very natural. "
        "It is not a great example of good writing."
    )
    result = h.humanize(text, strategy="rule-based")
    assert result.humanized_text
    assert result.strategy == "rule-based"
    print(f"  [OK] humanizer  (changes={len(result.changes_summary)}, "
          f"strategy={result.strategy})")
    for ch in result.changes_summary:
        print(f"       - {ch}")


def test_fingerprint():
    from core.fingerprint import FingerprintEngine
    engine = FingerprintEngine(profile_dir="fingerprints_test")
    texts = [
        "This is my first writing sample. I tend to use contractions like it's "
        "and don't. My sentences vary quite a bit in length. Some are short. "
        "Others stretch on for quite a while with complex subordinate clauses.",

        "Here is another sample of my writing. Again I'm using contractions. "
        "The sentence lengths still vary. Notice the informal tone throughout."
    ]
    profile = engine.create_profile(texts, "test_profile")
    assert profile.num_documents == 2
    assert len(profile.mean_vector) > 0

    result = engine.compare(texts[0], "test_profile")
    assert 0.0 <= result.similarity_score <= 1.0
    print(f"  [OK] fingerprint  (similarity={result.similarity_score:.2%}, "
          f"profile_docs={profile.num_documents})")

    # Cleanup
    engine.delete_profile("test_profile")
    import shutil, os
    if os.path.isdir("fingerprints_test"):
        shutil.rmtree("fingerprints_test")


def test_report_generator():
    from core.report_generator import generate_report
    from core.document_processor import DocumentMetadata
    from core.ai_detector import AIDetectionResult, PerplexityResult, StyleResult

    meta = DocumentMetadata(
        filename="test.txt", word_count=100, sentence_count=5,
        paragraph_count=2, char_count=500, avg_sentence_length=20.0,
    )
    ai = AIDetectionResult(
        ai_probability=0.65, confidence="Medium",
        perplexity_result=PerplexityResult(perplexity=42.5, normalised_score=0.6,
                                           interpretation="Moderate"),
        style_result=StyleResult(uniformity_score=0.5, vocabulary_score=0.3,
                                 naturalness_score=0.5),
        explanations=["Test explanation"],
        features={"burstiness": 0.4, "type_token_ratio": 0.55},
    )
    report = generate_report(meta, ai)
    assert "report_id" in report
    assert "ai_detection" in report
    assert "document_metadata" in report
    print(f"  [OK] report_generator  (report_id={report['report_id'][:8]}...)")


def test_db_layer():
    from db.chroma_client import get_collection, COLLECTION_CORPUS
    coll = get_collection(COLLECTION_CORPUS)
    assert coll is not None
    print(f"  [OK] db layer  (corpus collection ready, count={coll.count()})")


def main():
    tests = [
        ("Text Utils", test_text_utils),
        ("Document Processor", test_document_processor),
        ("Feature Extractor", test_feature_extractor),
        ("AI Detector", test_ai_detector),
        ("Plagiarism Detector", test_plagiarism_detector),
        ("Humanizer", test_humanizer),
        ("Fingerprint Engine", test_fingerprint),
        ("Report Generator", test_report_generator),
        ("DB Layer", test_db_layer),
    ]

    print("=" * 60)
    print("DocuGuard+ Integration Tests")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, func in tests:
        print(f"\n>> {name}")
        try:
            func()
            passed += 1
        except Exception:
            failed += 1
            traceback.print_exc()
            print(f"  [FAIL] {name}")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
