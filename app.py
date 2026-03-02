"""
app.py
------
Streamlit frontend for DocuGuard+.

Pages:
  1. Document Analysis  – Upload & analyse a document
  2. Corpus Management  – Add / view / remove reference documents
  3. Style Profiles     – Create / compare writing fingerprints
  4. Report History     – Browse past analysis reports
"""

from __future__ import annotations

import os
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DocuGuard+",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy engine singletons (cached per session to avoid repeated model loads)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading AI Detection Engine …")
def _load_ai_detector():
    from core.ai_detector import AIDetector
    return AIDetector()


@st.cache_resource(show_spinner="Loading Plagiarism Engine …")
def _load_plagiarism_detector():
    from core.plagiarism_detector import PlagiarismDetector
    return PlagiarismDetector()


def _load_humanizer():
    from core.humanizer import Humanizer
    return Humanizer()


@st.cache_resource(show_spinner=False)
def _load_fingerprint_engine():
    from core.fingerprint import FingerprintEngine
    return FingerprintEngine()


# ---------------------------------------------------------------------------
# Sidebar & navigation
# ---------------------------------------------------------------------------

def _sidebar():
    st.sidebar.title("🛡️ DocuGuard+")
    st.sidebar.caption(
        "Hybrid AI Detection, Semantic Plagiarism Analysis "
        "& Text Naturalization System"
    )
    page = st.sidebar.radio(
        "Navigate",
        ["📄 Document Analysis", "📚 Corpus Management",
         "✍️ Style Profiles", "📊 Report History"],
    )
    st.sidebar.divider()
    st.sidebar.info(
        "**Disclaimer:** AI detection is *probabilistic*, not definitive. "
        "Results are indicators, not proof of origin."
    )
    return page


# ═══════════════════════════════════════════════════════════════════════════
# Page 1: Document Analysis
# ═══════════════════════════════════════════════════════════════════════════

def _page_analysis():
    st.header("📄 Document Analysis")
    st.markdown(
        "Upload a document (PDF, DOCX, or TXT) **or** paste your text directly "
        "for AI detection, plagiarism checking, and optional humanization."
    )

    # ---- Input method tabs ----
    tab_upload, tab_paste = st.tabs(["📁 Upload File", "📋 Paste Text"])

    uploaded = None
    pasted_text = ""

    with tab_upload:
        uploaded = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "txt"],
            key="doc_upload",
        )

    with tab_paste:
        pasted_text = st.text_area(
            "Paste your text below",
            height=300,
            placeholder="Paste content from Google Docs, MS Word, or any source …",
            key="doc_paste",
        )
        paste_title = st.text_input(
            "Document title (optional)",
            value="Pasted Document",
            key="paste_title",
        )

        # ---- Live formatted preview ----
        if pasted_text.strip():
            with st.expander("📖 Formatted Preview", expanded=True):
                st.markdown(_plain_to_markdown(pasted_text), unsafe_allow_html=True)

    # ---- Analysis options ----
    col_opts1, col_opts2 = st.columns(2)
    with col_opts1:
        run_plagiarism = st.checkbox("Run plagiarism check", value=True)
    with col_opts2:
        run_fingerprint = st.checkbox("Compare with style profile", value=False)

    profile_name = "default"
    if run_fingerprint:
        engine = _load_fingerprint_engine()
        profiles = engine.list_profiles()
        if profiles:
            profile_name = st.selectbox("Select profile", profiles)
        else:
            st.warning("No style profiles found. Create one in the Style Profiles page.")
            run_fingerprint = False

    # ---- Determine which input to use ----
    has_input = uploaded is not None or pasted_text.strip()

    if has_input and st.button("🔍 Analyse Document", type="primary"):
        if uploaded is not None:
            _run_analysis_file(uploaded, run_plagiarism, run_fingerprint, profile_name)
        elif pasted_text.strip():
            _run_analysis_text(pasted_text.strip(), paste_title, run_plagiarism, run_fingerprint, profile_name)

    # --- Always render results from session state (survives reruns) ---
    if "last_report" in st.session_state:
        _display_results(
            st.session_state["last_report"],
            st.session_state["last_doc"],
            st.session_state["last_ai_result"],
            st.session_state.get("last_plag_result"),
            st.session_state.get("last_fp_result"),
        )


def _run_analysis_file(uploaded, run_plagiarism: bool, run_fingerprint: bool, profile_name: str):
    """Analyse from an uploaded file."""
    from core.document_processor import process_document

    file_bytes = uploaded.read()
    filename = uploaded.name

    with st.spinner("Extracting text …"):
        doc = process_document(file_bytes, filename)

    _run_analysis_common(doc, filename, run_plagiarism, run_fingerprint, profile_name)


def _run_analysis_text(text: str, title: str, run_plagiarism: bool, run_fingerprint: bool, profile_name: str):
    """Analyse from pasted text — build a DocumentResult directly."""
    from core.document_processor import DocumentResult, DocumentMetadata
    from utils.text_utils import clean_text, split_sentences, split_paragraphs, count_words

    with st.spinner("Processing text …"):
        cleaned = clean_text(text)
        sentences = split_sentences(cleaned)
        paragraphs = split_paragraphs(cleaned)
        wc = count_words(cleaned)

        filename = f"{title}.txt" if not title.endswith(".txt") else title
        meta = DocumentMetadata(
            filename=filename,
            word_count=wc,
            sentence_count=len(sentences),
            paragraph_count=len(paragraphs),
            char_count=len(cleaned),
            avg_sentence_length=wc / len(sentences) if sentences else 0.0,
        )
        doc = DocumentResult(
            raw_text=text,
            cleaned_text=cleaned,
            sentences=sentences,
            paragraphs=paragraphs,
            metadata=meta,
        )

    _run_analysis_common(doc, filename, run_plagiarism, run_fingerprint, profile_name)


def _run_analysis_common(doc, filename: str, run_plagiarism: bool, run_fingerprint: bool, profile_name: str):
    """Shared analysis pipeline for both file-upload and pasted-text inputs."""
    from core.report_generator import generate_report, save_report
    from db.history_store import add_analysis

    if doc.metadata.word_count == 0:
        st.error("No text could be extracted from the file.")
        return

    if doc.metadata.word_count > 15_000:
        st.warning(
            f"Document has {doc.metadata.word_count:,} words. "
            "Analysis is optimised for ≤ 15,000 words and may be slower."
        )

    # --- Step 2: AI Detection ---
    with st.spinner("Running AI detection …"):
        detector = _load_ai_detector()
        ai_result = detector.detect(doc.cleaned_text)

    # --- Step 3: Plagiarism (optional) ---
    plag_result = None
    if run_plagiarism:
        with st.spinner("Running plagiarism check …"):
            plag = _load_plagiarism_detector()
            plag_result = plag.check(doc.cleaned_text)

    # --- Step 4: Fingerprint (optional) ---
    fp_result = None
    if run_fingerprint:
        with st.spinner("Comparing style profile …"):
            fp_engine = _load_fingerprint_engine()
            fp_result = fp_engine.compare(doc.cleaned_text, profile_name)

    # --- Generate & persist report ---
    report = generate_report(
        metadata=doc.metadata,
        ai_result=ai_result,
        plagiarism_result=plag_result,
        fingerprint_result=fp_result,
    )
    report_path = save_report(report)

    add_analysis(
        report_id=report["report_id"],
        filename=filename,
        plagiarism_score=plag_result.overall_score if plag_result else 0.0,
        ai_score=ai_result.ai_probability,
        verdict=_verdict_label(ai_result.ai_probability),
        matched_sources=plag_result.matched_sources if plag_result else [],
        fingerprint_similarity=fp_result.similarity_score if fp_result else None,
    )

    # --- Store ALL results in session for re-rendering & humanization ---
    st.session_state["last_doc"] = doc
    st.session_state["last_ai_result"] = ai_result
    st.session_state["last_report"] = report
    st.session_state["last_plag_result"] = plag_result
    st.session_state["last_fp_result"] = fp_result


# ---------------------------------------------------------------------------
# Helper: inline diff rendering
# ---------------------------------------------------------------------------
def _render_inline_diff(original: str, humanized: str):
    """Show a sentence-level inline diff using difflib."""
    import difflib, html as _html

    orig_sents = [s.strip() for s in original.replace("\n", " ").split(". ") if s.strip()]
    hum_sents = [s.strip() for s in humanized.replace("\n", " ").split(". ") if s.strip()]
    matcher = difflib.SequenceMatcher(None, orig_sents, hum_sents)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for s in orig_sents[i1:i2]:
                parts.append(_html.escape(s) + ".")
        elif tag == "replace":
            for s in orig_sents[i1:i2]:
                parts.append(f'<span style="background:#ffcccc;text-decoration:line-through">{_html.escape(s)}.</span>')
            for s in hum_sents[j1:j2]:
                parts.append(f'<span style="background:#ccffcc">{_html.escape(s)}.</span>')
        elif tag == "delete":
            for s in orig_sents[i1:i2]:
                parts.append(f'<span style="background:#ffcccc;text-decoration:line-through">{_html.escape(s)}.</span>')
        elif tag == "insert":
            for s in hum_sents[j1:j2]:
                parts.append(f'<span style="background:#ccffcc">{_html.escape(s)}.</span>')
    st.markdown(" ".join(parts), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: generate PDF report bytes (HTML-based via simple markup)
# ---------------------------------------------------------------------------
def _generate_pdf_report(report: dict) -> bytes:
    """Build a lightweight PDF from the analysis report using fpdf2."""
    try:
        from fpdf import FPDF
    except ImportError:
        # Fallback: return an HTML file encoded as bytes if fpdf2 missing
        return _generate_html_report_bytes(report)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "DocuGuard+ Analysis Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Report ID: {report.get('report_id', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Generated: {report.get('timestamp', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Metadata section
    meta = report.get("metadata", {})
    if meta:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Document Metadata", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for key in ("filename", "word_count", "char_count", "language"):
            val = meta.get(key, "N/A")
            pdf.cell(0, 6, f"  {key}: {val}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    # AI Detection section
    ai = report.get("ai_detection", {})
    if ai:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "AI Detection", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"  AI Probability: {ai.get('ai_probability', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, f"  Confidence: {ai.get('confidence', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, f"  Mean Perplexity: {ai.get('mean_perplexity', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    # Plagiarism section
    plag = report.get("plagiarism", {})
    if plag:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Plagiarism Analysis", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"  Overall Score: {plag.get('overall_score', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
        sources = plag.get("matched_sources", [])
        if sources:
            pdf.cell(0, 6, f"  Matched Sources: {len(sources)}", new_x="LMARGIN", new_y="NEXT")
            for src in sources[:10]:
                if isinstance(src, dict):
                    pdf.cell(0, 6, f"    - {src.get('source', 'unknown')} ({src.get('similarity', 0):.0%})", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    # Fingerprint section
    fp = report.get("fingerprint", {})
    if fp:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Style Fingerprint", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"  Similarity: {fp.get('similarity_score', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    return bytes(pdf.output())


def _generate_html_report_bytes(report: dict) -> bytes:
    """Fallback: generate an HTML report when fpdf2 is not installed."""
    import json as _json
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>DocuGuard+ Report</title>
<style>body{{font-family:sans-serif;margin:2em}}h1{{color:#1a237e}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:6px 10px;text-align:left}}
th{{background:#e8eaf6}}</style></head><body>
<h1>DocuGuard+ Analysis Report</h1>
<p><b>Report ID:</b> {report.get('report_id','N/A')}</p>
<p><b>Generated:</b> {report.get('timestamp','N/A')}</p>
<h2>Full Report Data</h2>
<pre>{_json.dumps(report, indent=2, default=str)}</pre>
</body></html>"""
    return html.encode("utf-8")


def _display_results(report, doc, ai_result, plag_result, fp_result):
    st.divider()
    st.subheader("Analysis Results")

    # ---- Metrics row ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Words", f"{doc.metadata.word_count:,}")
    c2.metric("AI Probability", f"{ai_result.ai_probability:.0%}")
    c3.metric("Confidence", ai_result.confidence)
    if plag_result:
        c4.metric("Plagiarism Score", f"{plag_result.overall_score:.0%}")
    else:
        c4.metric("Plagiarism Score", "N/A")

    # ---- AI Detection Tab ----
    tab_ai, tab_plag, tab_fp, tab_human = st.tabs(
        ["🤖 AI Detection", "📋 Plagiarism", "✍️ Fingerprint", "🔄 Humanize"]
    )

    with tab_ai:
        _render_ai_tab(ai_result, report)

    with tab_plag:
        _render_plagiarism_tab(plag_result)

    with tab_fp:
        _render_fingerprint_tab(fp_result)

    with tab_human:
        _render_humanize_tab()

    # ---- Report download buttons ----
    st.divider()
    dl_cols = st.columns(2)
    import json as _json
    report_json = _json.dumps(report, indent=2, default=str)
    dl_cols[0].download_button(
        "📥 Download JSON Report",
        data=report_json,
        file_name=f"docuguard_report_{report.get('report_id', 'unknown')}.json",
        mime="application/json",
    )
    # PDF export
    pdf_bytes = _generate_pdf_report(report)
    dl_cols[1].download_button(
        "📄 Download PDF Report",
        data=pdf_bytes,
        file_name=f"docuguard_report_{report.get('report_id', 'unknown')}.pdf",
        mime="application/pdf",
    )


def _render_ai_tab(ai_result, report):
    # Probability gauge
    prob = ai_result.ai_probability
    if prob >= 0.75:
        colour = "🔴"
    elif prob >= 0.45:
        colour = "🟡"
    else:
        colour = "🟢"
    st.markdown(f"### {colour} AI Probability: **{prob:.1%}** ({ai_result.confidence} confidence)")

    # Explanations
    st.markdown("#### Explanations")
    for exp in ai_result.explanations:
        st.markdown(f"- {exp}")

    # Key features
    st.markdown("#### Key Features")
    key_feats = report.get("ai_detection", {}).get("key_features", {})
    if key_feats:
        cols = st.columns(min(len(key_feats), 5))
        for i, (k, v) in enumerate(key_feats.items()):
            with cols[i % len(cols)]:
                label = k.replace("_", " ").title()
                st.metric(label, f"{v:.3f}" if isinstance(v, float) else str(v))

    # Perplexity detail
    with st.expander("Perplexity Details"):
        ppl = ai_result.perplexity_result
        st.write(f"**Perplexity:** {ppl.perplexity:.2f}")
        st.write(f"**Mean token loss:** {ppl.mean_token_loss:.4f}")
        st.write(f"**Normalised score:** {ppl.normalised_score:.4f}")
        st.write(f"**Interpretation:** {ppl.interpretation}")

    # Style detail
    with st.expander("Stylometric Details"):
        sty = ai_result.style_result
        st.write(f"**Uniformity:** {sty.uniformity_score:.3f}")
        st.write(f"**Vocabulary score:** {sty.vocabulary_score:.3f}")
        st.write(f"**Naturalness:** {sty.naturalness_score:.3f}")
        for f in sty.contributing_factors:
            st.markdown(f"- {f}")

    # Per-paragraph breakdown (heatmap)
    if hasattr(ai_result, "paragraph_scores") and ai_result.paragraph_scores:
        with st.expander("📊 Per-Paragraph AI Breakdown", expanded=True):
            st.caption(
                "Each paragraph is scored independently.  "
                "Red = likely AI, green = likely human."
            )
            for ps in ai_result.paragraph_scores:
                prob_p = ps.ai_probability
                if prob_p >= 0.75:
                    badge = "🔴"
                elif prob_p >= 0.45:
                    badge = "🟡"
                else:
                    badge = "🟢"
                preview = ps.text_preview.replace("\n", " ")
                if len(preview) > 70:
                    preview = preview[:70] + "…"
                st.markdown(
                    f"{badge} **¶{ps.index + 1}** — AI: **{prob_p:.0%}** · "
                    f"PPL: {ps.perplexity:.0f} · {ps.word_count} words  \n"
                    f"<span style='color:gray;font-size:0.85em'>{preview}</span>",
                    unsafe_allow_html=True,
                )


def _render_plagiarism_tab(plag_result):
    if plag_result is None:
        st.info("Plagiarism check was not run.")
        return

    st.markdown(
        f"### Overall Plagiarism Score: **{plag_result.overall_score:.1%}**"
    )
    c1, c2 = st.columns(2)
    c1.metric("Lexical Score", f"{plag_result.lexical_score:.1%}")
    c2.metric("Semantic Score", f"{plag_result.semantic_score:.1%}")

    if plag_result.matched_sources:
        st.markdown("**Matched sources:**")
        for src in plag_result.matched_sources:
            st.markdown(f"- `{src}`")

    if plag_result.matched_segments:
        st.markdown("#### Flagged Segments")
        for seg in plag_result.matched_segments[:15]:
            with st.expander(
                f"[{seg.match_type.upper()}] Similarity {seg.similarity:.0%} — {seg.source_filename}"
            ):
                st.markdown("**Your text:**")
                st.text(seg.input_text[:300])
                st.markdown("**Matched corpus text:**")
                st.text(seg.matched_text[:300])

    st.caption(plag_result.disclaimer)


def _render_fingerprint_tab(fp_result):
    if fp_result is None:
        st.info("Style-profile comparison was not run.")
        return

    if fp_result.similarity_score == 0.0 and not fp_result.deviation_summary:
        st.warning(fp_result.interpretation)
        return

    st.markdown(
        f"### Style Similarity: **{fp_result.similarity_score:.1%}** "
        f"(profile: *{fp_result.profile_name}*)"
    )
    st.write(fp_result.interpretation)

    if fp_result.deviation_summary:
        st.markdown("#### Top Deviations from Profile")
        sorted_devs = sorted(
            fp_result.deviation_summary.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:8]
        for k, v in sorted_devs:
            direction = "⬆️" if v > 0 else "⬇️"
            st.markdown(f"- **{k.replace('_', ' ').title()}**: {direction} {v:+.4f}")


def _render_humanize_tab():
    st.markdown("### Controlled Text Humanization")
    st.caption(
        "Improve naturalness and variability. This is a writing-improvement "
        "tool, NOT a detection-evasion tool."
    )

    if "last_doc" not in st.session_state:
        st.info("Analyse a document first to enable humanization.")
        return

    strategy = st.radio(
        "Strategy",
        ["rule-based", "llm"],
        captions=[
            "Fast, deterministic, offline",
            "Uses Ollama (requires running Ollama service)",
        ],
    )

    # Show model selector when LLM strategy is chosen
    model_name = "mistral"
    h_config = None
    output_format = "plain"  # Default for rule-based strategy

    if strategy == "llm":
        import requests as _req
        from core.humanizer import HumanizationConfig, MAX_WORD_LIMIT

        _ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        # Dynamically fetch available models from Ollama
        _available_models: list[str] = []
        try:
            _r = _req.get(f"{_ollama_url}/api/tags", timeout=3)
            _available_models = [
                m["name"].split(":")[0]
                for m in _r.json().get("models", [])
            ]
            _seen: set[str] = set()
            _available_models = [
                m for m in _available_models
                if not (m in _seen or _seen.add(m))  # type: ignore[func-returns-value]
            ]
        except Exception:
            pass

        if not _available_models:
            _available_models = ["mistral", "llama3"]
            st.error(f"Cannot reach Ollama at {_ollama_url}. Is the container running?")
        else:
            st.success(f"Ollama connected — {len(_available_models)} model(s) available")

        model_name = st.selectbox(
            "LLM Model",
            _available_models,
            help="Select which Ollama model to use for rewriting.",
        )

        # ── User-controllable humanization settings ──────────────────
        with st.expander("⚙️ Humanization Settings", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                tone = st.selectbox(
                    "Writing Tone",
                    ["academic", "casual", "professional", "creative"],
                    index=0,
                    help="Controls the voice and register of the rewritten text.",
                )
                intensity = st.select_slider(
                    "Rewrite Intensity",
                    options=["light", "balanced", "aggressive"],
                    value="balanced",
                    help=(
                        "Light: minimal changes, preserve voice. "
                        "Balanced: substantial rewrite. "
                        "Aggressive: deep restructuring."
                    ),
                )
            with col_b:
                domain = st.text_input(
                    "Subject Domain (optional)",
                    placeholder="e.g. computer science, history, biology",
                    help="Helps the LLM use domain-appropriate vocabulary.",
                )
                preserve_kw = st.text_input(
                    "Preserve Keywords (optional)",
                    placeholder="e.g. BERT, transformer, fine-tuning",
                    help="Comma-separated terms that must appear verbatim.",
                )

            custom_instr = st.text_area(
                "Custom Instructions (optional)",
                placeholder="e.g. 'Keep the conclusion paragraph intact' or 'Use British English'",
                height=68,
                help="Free-form notes passed directly to the LLM.",
            )

            col_c, col_d = st.columns(2)
            with col_c:
                target_prob = st.slider(
                    "Target AI Probability",
                    min_value=0.10, max_value=0.50, value=0.30, step=0.05,
                    help="Pipeline stops iterating once AI prob drops below this.",
                )
            with col_d:
                max_passes = st.slider(
                    "Max LLM Passes",
                    min_value=1, max_value=3, value=3,
                    help="More passes = better results but longer processing time.",
                )

            output_format = st.radio(
                "Output Format",
                options=["plain", "markdown"],
                index=0,
                horizontal=True,
                help="Markdown adds headers, bold, and lists for better readability.",
            )

        # Show word-count info
        _wc = len(st.session_state["last_doc"].cleaned_text.split())
        if _wc > MAX_WORD_LIMIT:
            st.warning(
                f"Input is {_wc:,} words — only the first {MAX_WORD_LIMIT:,} "
                f"words will be processed; the rest is appended unchanged."
            )
        else:
            st.info(f"Input: {_wc:,} words (limit: {MAX_WORD_LIMIT:,})")

        h_config = HumanizationConfig(
            tone=tone,
            intensity=intensity,
            domain=domain.strip(),
            preserve_keywords=preserve_kw.strip(),
            custom_instructions=custom_instr.strip(),
            target_ai_prob=target_prob,
            max_passes=max_passes,
            output_format=output_format,
        )

    if st.button("🔄 Humanize & Re-evaluate", type="primary"):
        import time as _time
        doc = st.session_state["last_doc"]
        ai_before = st.session_state["last_ai_result"]

        humanizer = _load_humanizer()
        start_time = _time.time()

        # --- Progress UI for LLM pipeline ---------------------------------
        if strategy == "llm":
            progress_bar = st.progress(0.0, text="Initialising LLM pipeline …")
            status_text = st.empty()
            timer_display = st.empty()

            def _progress_cb(stage: str, pct: float):
                elapsed = _time.time() - start_time
                progress_bar.progress(min(pct, 1.0), text=stage)
                status_text.caption(stage)
                timer_display.markdown(f"⏱️ **Elapsed:** {elapsed:.1f}s")

            h_result = humanizer.humanize(
                doc.cleaned_text,
                strategy=strategy,
                model=model_name,
                progress_cb=_progress_cb,
                config=h_config,
            )
            elapsed_total = _time.time() - start_time
            h_result.elapsed_seconds = elapsed_total
            progress_bar.progress(1.0, text="✅ Humanization complete")
            timer_display.markdown(f"✅ **Completed in {elapsed_total:.1f}s**")
            status_text.empty()
        else:
            with st.spinner("Humanizing text …"):
                h_result = humanizer.humanize(doc.cleaned_text, strategy=strategy, model=model_name)
                h_result.elapsed_seconds = _time.time() - start_time

        with st.spinner("Final AI evaluation …"):
            detector = _load_ai_detector()
            ai_after = detector.detect(h_result.humanized_text)

        h_result.ai_prob_before = ai_before.ai_probability
        h_result.ai_prob_after = ai_after.ai_probability

        # Persist humanization results in session state
        st.session_state["last_humanize_result"] = h_result
        st.session_state["last_humanize_ai_before"] = ai_before
        st.session_state["last_humanize_ai_after"] = ai_after
        st.session_state["last_humanize_format"] = output_format

    # --- Always render results from session state -------------------------
    if "last_humanize_result" in st.session_state:
        h_result = st.session_state["last_humanize_result"]
        ai_before = st.session_state["last_humanize_ai_before"]
        ai_after = st.session_state["last_humanize_ai_after"]

        st.divider()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("AI Prob. (Before)", f"{ai_before.ai_probability:.1%}")
        c2.metric("AI Prob. (After)", f"{ai_after.ai_probability:.1%}")
        delta = ai_before.ai_probability - ai_after.ai_probability
        c3.metric("Improvement", f"{delta:+.1%}")
        c4.metric("LLM Passes", h_result.passes_used if hasattr(h_result, "passes_used") else "—")
        elapsed = getattr(h_result, "elapsed_seconds", 0)
        if elapsed >= 60:
            c5.metric("Time", f"{elapsed / 60:.1f}m")
        else:
            c5.metric("Time", f"{elapsed:.1f}s")

        if hasattr(h_result, "word_count_original") and h_result.word_count_original:
            st.caption(
                f"Words: {h_result.word_count_original:,} → "
                f"{h_result.word_count_humanized:,}"
            )

        st.markdown("**Pipeline stages:**")
        for ch in h_result.changes_summary:
            st.markdown(f"- {ch}")

        with st.expander("View Humanized Text", expanded=True):
            # Render as Markdown if that format was selected
            if st.session_state.get("last_humanize_format") == "markdown":
                st.markdown(h_result.humanized_text)
            else:
                st.text_area("Humanized output", h_result.humanized_text, height=300)

        # Side-by-side diff view
        with st.expander("📝 Side-by-Side Comparison"):
            diff_col_a, diff_col_b = st.columns(2)
            with diff_col_a:
                st.markdown("**Original**")
                st.text_area(
                    "Original text",
                    h_result.original_text,
                    height=300,
                    key="diff_original",
                    label_visibility="collapsed",
                )
            with diff_col_b:
                st.markdown("**Humanized**")
                st.text_area(
                    "Humanized text",
                    h_result.humanized_text,
                    height=300,
                    key="diff_humanized",
                    label_visibility="collapsed",
                )
            # Inline diff (highlighted sentence-level changes)
            _render_inline_diff(h_result.original_text, h_result.humanized_text)

        # Download button with appropriate extension
        is_md = st.session_state.get("last_humanize_format") == "markdown"
        st.download_button(
            "📋 Download Humanized Text",
            data=h_result.humanized_text,
            file_name="humanized_output.md" if is_md else "humanized_output.txt",
            mime="text/markdown" if is_md else "text/plain",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Page 2: Corpus Management
# ═══════════════════════════════════════════════════════════════════════════

def _page_corpus():
    st.header("📚 Corpus Management")
    st.markdown(
        "Manage the local reference corpus used for plagiarism detection. "
        "Upload reference documents to compare against."
    )

    # Upload
    uploaded_files = st.file_uploader(
        "Upload reference documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="corpus_upload",
    )

    if uploaded_files and st.button("➕ Add to Corpus", type="primary"):
        from core.document_processor import process_document
        from db.corpus_store import add_document

        for uf in uploaded_files:
            with st.spinner(f"Processing {uf.name} …"):
                try:
                    doc = process_document(uf.read(), uf.name)
                    add_document(doc.cleaned_text, uf.name)
                    st.success(f"Added **{uf.name}** ({doc.metadata.word_count:,} words)")
                except Exception as e:
                    st.error(f"Failed to add {uf.name}: {e}")

    # List existing
    st.divider()
    st.subheader("Current Corpus")
    from db.corpus_store import list_documents, delete_document, corpus_count

    docs = list_documents()
    st.caption(f"Total chunks: {corpus_count()}")

    if not docs:
        st.info("Corpus is empty. Upload reference documents above.")
    else:
        for d in docs:
            col1, col2, col3 = st.columns([4, 2, 1])
            col1.write(f"**{d.get('filename', 'Unknown')}**")
            col2.write(f"{d.get('total_chunks', '?')} chunks")
            if col3.button("🗑️", key=f"del_{d['doc_id']}"):
                delete_document(d["doc_id"])
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# Page 3: Style Profiles
# ═══════════════════════════════════════════════════════════════════════════

def _page_profiles():
    st.header("✍️ Style Profiles")
    st.markdown(
        "Upload your own past work to create a personal writing-style "
        "fingerprint. New documents can be compared against it."
    )

    engine = _load_fingerprint_engine()

    # Create profile
    st.subheader("Create / Update Profile")
    profile_name = st.text_input("Profile name", value="default")
    uploaded = st.file_uploader(
        "Upload your own past documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="profile_upload",
    )

    if uploaded and st.button("🔨 Build Profile", type="primary"):
        from core.document_processor import process_document

        texts: list[str] = []
        for uf in uploaded:
            with st.spinner(f"Processing {uf.name} …"):
                try:
                    doc = process_document(uf.read(), uf.name)
                    if doc.cleaned_text:
                        texts.append(doc.cleaned_text)
                except Exception as e:
                    st.error(f"Failed to process {uf.name}: {e}")

        if texts:
            profile = engine.create_profile(texts, profile_name)
            st.success(
                f"Profile **{profile_name}** created from "
                f"{profile.num_documents} document(s)."
            )

    # List profiles
    st.divider()
    st.subheader("Saved Profiles")
    profiles = engine.list_profiles()
    if not profiles:
        st.info("No profiles yet. Upload documents above to create one.")
    else:
        for pname in profiles:
            col1, col2 = st.columns([5, 1])
            col1.write(f"📝 **{pname}**")
            if col2.button("🗑️", key=f"delprofile_{pname}"):
                engine.delete_profile(pname)
                st.rerun()

            # Show profile stats
            profile = engine.load_profile(pname)
            if profile:
                with st.expander(f"Details: {pname}"):
                    st.write(f"Documents used: {profile.num_documents}")
                    st.write("Mean features:")
                    st.json(profile.mean_features)


# ═══════════════════════════════════════════════════════════════════════════
# Page 4: Report History
# ═══════════════════════════════════════════════════════════════════════════

def _page_history():
    st.header("📊 Report History")

    from core.report_generator import list_reports, load_report
    from db.history_store import analysis_count

    st.caption(f"Total analyses: {analysis_count()}")

    reports = list_reports()
    if not reports:
        st.info("No reports yet. Analyse a document to create one.")
        return

    for r in reports:
        with st.expander(
            f"📄 {r.get('filename', 'Unknown')} — "
            f"AI: {r.get('ai_probability', '?')} — "
            f"{r.get('generated_at', '')[:19]}"
        ):
            full_report = load_report(r["report_id"])
            if full_report:
                st.json(full_report)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verdict_label(prob: float) -> str:
    if prob >= 0.75:
        return "likely_ai"
    if prob >= 0.45:
        return "mixed"
    return "likely_human"


def _plain_to_markdown(text: str) -> str:
    """Convert plain text (pasted from Google Docs / Word) into rich HTML.

    Uses HTML rather than pure Markdown so that every line break is respected
    exactly as the user pasted it — matching the look of Google Docs / Word.

    Heuristics:
      - First non-blank line → large title
      - Short standalone lines (≤ 10 words, no period) → section headings
      - "3.1 …" lines → indented bold-numbered sub-items
      - "3. …" lines → bold numbered sections
      - Bullet chars → list items
      - Regular text → paragraphs
      - Blank lines → visual paragraph breaks
    """
    import re

    lines = text.split("\n")
    html_parts: list[str] = []
    title_emitted = False
    prev_blank = True

    for raw_line in lines:
        line = raw_line.rstrip()

        # Blank line → spacer
        if not line.strip():
            html_parts.append('<div style="height:0.6em"></div>')
            prev_blank = True
            continue

        stripped = line.strip()
        words = stripped.split()
        is_short = len(words) <= 10
        no_period = not stripped.endswith((".", ",", ";", ":"))
        has_sub_num = bool(re.match(r"^\d+\.\d+", stripped))
        has_chap_num = bool(re.match(r"^\d+\.\s", stripped))

        # ---- Numbered sub-section "3.1 Title" ----
        m_sub = re.match(r"^(\d+\.\d+)\s+(.+)$", stripped)
        if m_sub and len(words) <= 12 and no_period:
            html_parts.append(
                f'<div style="padding-left:2em;margin:2px 0">'
                f'<b>{m_sub.group(1)}</b>&ensp;{_esc(m_sub.group(2))}</div>'
            )
            prev_blank = False
            continue

        # ---- Chapter numbering "3. Introduction" ----
        m_chap = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if m_chap and len(words) <= 8 and no_period:
            html_parts.append(
                f'<div style="margin:6px 0;font-weight:600">'
                f'{m_chap.group(1)}. {_esc(m_chap.group(2))}</div>'
            )
            prev_blank = False
            continue

        # ---- Heading detection ----
        if is_short and no_period and not has_sub_num and not has_chap_num:
            if not title_emitted:
                html_parts.append(
                    f'<h2 style="margin:0.2em 0 0.3em">{_esc(stripped)}</h2>'
                )
                title_emitted = True
                prev_blank = False
                continue

            if prev_blank:
                html_parts.append(
                    f'<h4 style="margin:0.5em 0 0.2em">{_esc(stripped)}</h4>'
                )
                prev_blank = False
                continue

            # Short line not after blank → TOC-style entry
            html_parts.append(
                f'<div style="margin:1px 0"><b>{_esc(stripped)}</b></div>'
            )
            prev_blank = False
            continue

        # ---- Bullet chars ----
        if re.match(r"^[-*•○◦]\s", stripped):
            html_parts.append(
                f'<div style="padding-left:1.5em;margin:2px 0">'
                f'• {_esc(stripped[2:].strip())}</div>'
            )
            prev_blank = False
            continue

        # ---- Regular paragraph ----
        html_parts.append(f'<p style="margin:0.3em 0">{_esc(stripped)}</p>')
        prev_blank = False

    return "\n".join(html_parts)


def _esc(text: str) -> str:
    """Minimal HTML escaping for user text."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    page = _sidebar()

    if page == "📄 Document Analysis":
        _page_analysis()
    elif page == "📚 Corpus Management":
        _page_corpus()
    elif page == "✍️ Style Profiles":
        _page_profiles()
    elif page == "📊 Report History":
        _page_history()


if __name__ == "__main__":
    main()
