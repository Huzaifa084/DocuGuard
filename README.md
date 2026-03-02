# 🛡️ DocuGuard+

**Hybrid AI Detection, Semantic Plagiarism Analysis & Text Naturalization System**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 What is DocuGuard+?

DocuGuard+ is an all-in-one document analysis platform that combines:

- **🤖 AI Detection** — Hybrid engine using perplexity analysis + stylometry + ML classifier to detect AI-generated text
- **📋 Plagiarism Detection** — Semantic similarity search against a local corpus using sentence embeddings + TF-IDF
- **✍️ Style Fingerprinting** — Create and compare writing style profiles using 12+ stylometric features
- **🔄 LLM Humanization** — Multi-pass intelligent text rewriting that makes AI-generated text read naturally human

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Frontend                     │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌───────────┐ │
│  │ Document  │ │  Corpus   │ │  Style   │ │  Report   │ │
│  │ Analysis  │ │ Management│ │ Profiles │ │  History  │ │
│  └────┬─────┘ └─────┬─────┘ └────┬─────┘ └─────┬─────┘ │
└───────┼─────────────┼────────────┼──────────────┼───────┘
        │             │            │              │
┌───────▼─────────────▼────────────▼──────────────▼───────┐
│                    Core Engines                          │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐  │
│  │ AI Detector   │  │  Plagiarism   │  │ Fingerprint  │  │
│  │ (DistilGPT-2  │  │  Detector     │  │   Engine     │  │
│  │  + StyleNet)  │  │ (ChromaDB +   │  │ (12+ feat.)  │  │
│  └──────────────┘  │  Embeddings)  │  └──────────────┘  │
│                    └───────────────┘                     │
│  ┌──────────────────────────────────────────────────┐   │
│  │              LLM Humanizer                        │   │
│  │  Pre-analysis → Paragraph Rewrite → Sentence Fix  │   │
│  │  → Detection Gate → Iterative Passes → Polish     │   │
│  └──────────────────┬───────────────────────────────┘   │
└─────────────────────┼───────────────────────────────────┘
                      │
         ┌────────────▼────────────┐
         │    Ollama LLM Server    │
         │  (Local or Colab GPU)   │
         │  Mistral / LLama 3     │
         └─────────────────────────┘
```

---

## ✨ Key Features

### AI Detection Engine
- **Sliding-window perplexity** analysis using DistilGPT-2 (batched inference for speed)
- **12+ stylometric features**: burstiness, TTR, passive voice, sentence starter diversity, etc.
- **ML classifier** combining perplexity + style signals
- **Per-paragraph AI heatmap** — color-coded breakdown showing which paragraphs are most AI-like

### Plagiarism Detector
- **Semantic search** via sentence-transformers (`all-MiniLM-L6-v2`) + ChromaDB
- **TF-IDF lexical matching** as secondary check
- **Coverage-based scoring** — measures what fraction of your document matches corpus sources

### LLM Humanizer (Multi-Pass Pipeline)
- **Pre-analysis** — detects specific AI patterns before rewriting
- **Context-aware paragraph rewriting** — each paragraph rewritten with awareness of surrounding context
- **Sentence-level targeted fixes** — identifies and rewrites the most AI-like individual sentences
- **Iterative detection gate** — re-evaluates after each pass, stops when target AI probability reached
- **Dynamic system prompts** — tailored to user's tone, intensity, domain preferences
- **Markdown output format** — clean, structured output with headings, bold, lists
- **Real-time progress bar & timer** — live feedback showing elapsed time and current stage

### UI Features
- **Remote LLM support** — paste a Google Colab Cloudflare tunnel URL in the sidebar
- **Side-by-side diff view** — compare original vs humanized text
- **JSON + PDF report export**
- **Style profile management** — create, compare, and store writing fingerprints

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) (local) **or** Google Colab (remote GPU)
- 8 GB RAM minimum (for AI detection models)

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/DocuGuard.git
cd DocuGuard

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up LLM Backend

**Option A: Local Ollama (CPU)**
```bash
# Install Ollama from https://ollama.com
ollama pull mistral
ollama serve
```

**Option B: Docker**
```bash
docker-compose up -d
```

**Option C: Google Colab GPU (Recommended for speed)**

Open [`docuguard_colab_gpu.ipynb`](docuguard_colab_gpu.ipynb) in Google Colab, run all cells, and paste the tunnel URL into the sidebar. See [Colab GPU Setup](#-colab-gpu-setup) below.

### 3. Run the App

```bash
streamlit run app.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🖥️ Usage Flow

### Document Analysis

```
Upload Document (.pdf/.docx/.txt)  or  Paste Text
        │
        ▼
   ┌─────────────┐
   │  Document    │  Extracts text, metadata, word count
   │  Processor   │
   └──────┬──────┘
          │
    ┌─────▼─────┐──────────────┐───────────────┐
    │           │              │               │
    ▼           ▼              ▼               ▼
 🤖 AI      📋 Plagiarism  ✍️ Style      🔄 Humanize
 Detection   Check          Fingerprint    (LLM)
    │           │              │               │
    └─────┬─────┘──────────────┘───────────────┘
          │
          ▼
   📊 Results Dashboard
   • Metrics row (AI prob, plagiarism score, word count)
   • Per-paragraph AI heatmap
   • Matched sources list
   • Side-by-side diff view
   • Download JSON / PDF report
```

### Humanization Pipeline (LLM Strategy)

```
Input Text
    │
    ▼
1. Pre-Analysis ─── Run AI detector, identify specific patterns
    │
    ▼
2. Build Dynamic System Prompt ─── Tailored to tone/intensity/domain
    │
    ▼
3. Context-Aware Paragraph Rewriting ─── Each paragraph rewritten
   │                                      with prev/next context
   ▼
4. AI Detection Gate ─── Check if AI prob < target
   │
   ├─ YES → Skip to Polish
   │
   ├─ NO → Sentence-Level Fixes ─── Fix worst AI-like sentences
   │        │
   │        ▼
   │       Full Targeted Rewrite ─── Guided by detection diagnosis
   │        │
   │        ▼
   └─── Loop (up to max_passes) ──→ Back to Gate
    │
    ▼
5. Rule-Based Polish ─── Contractions, vocabulary, transitions
    │
    ▼
Output (Plain Text or Markdown)
```

---

## ☁️ Colab GPU Setup

For **5-8x faster** humanization using Google Colab's free T4 GPU:

1. Open [`docuguard_colab_gpu.ipynb`](docuguard_colab_gpu.ipynb) in [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU**: `Runtime → Change runtime type → T4 GPU`
3. Run all cells sequentially
4. Copy the Cloudflare tunnel URL displayed in Step 5
5. Paste the URL into DocuGuard+ sidebar → **🔗 LLM Backend** field

| Setup | Inference Speed | 500-word doc |
|-------|----------------|--------------|
| Local CPU | ~5-15 tok/s | ~2 min |
| **Colab T4** | **~40-80 tok/s** | **~15-25 sec** |

> **Note:** The tunnel URL changes each time you restart the notebook. Keep the notebook running during your session.

---

## 📁 Project Structure

```
DocuGuard/
├── app.py                      # Streamlit frontend (all pages)
├── core/
│   ├── ai_detector.py          # Hybrid AI detection engine
│   ├── plagiarism_detector.py  # Semantic + lexical plagiarism
│   ├── humanizer.py            # Multi-pass LLM humanization
│   ├── fingerprint.py          # Style fingerprinting engine
│   ├── feature_extractor.py    # 12+ stylometric features
│   ├── document_processor.py   # PDF/DOCX/TXT text extraction
│   └── report_generator.py     # JSON report generation
├── db/
│   ├── chroma_client.py        # ChromaDB singleton
│   ├── corpus_store.py         # Reference corpus management
│   └── history_store.py        # SQLite analysis history
├── utils/
│   └── text_utils.py           # Sentence splitting, helpers
├── docuguard_colab_gpu.ipynb   # Google Colab GPU notebook
├── docker-compose.yml          # Docker setup (app + Ollama)
├── Dockerfile                  # App container
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## ⚙️ Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint (or Colab tunnel URL) |
| `DEFAULT_OLLAMA_MODEL` | `mistral` | Default LLM model for humanization |
| `HISTORY_DB_PATH` | `data/history.db` | SQLite database path for report history |
| `TRANSFORMERS_CACHE` | (system default) | Model cache directory |

---

## 🧪 Humanization Settings

| Setting | Options | Description |
|---------|---------|-------------|
| **Tone** | `academic` / `casual` / `professional` / `creative` | Writing voice and register |
| **Intensity** | `light` / `balanced` / `aggressive` | How much to restructure |
| **Output Format** | `markdown` / `plain` | Markdown for rich formatting |
| **Target AI Prob** | 0.10 – 0.50 | Stop when AI probability drops below this |
| **Max Passes** | 1 – 3 | Maximum LLM rewriting iterations |
| **Domain** | Free text | Subject area for vocabulary tuning |
| **Preserve Keywords** | Comma-separated | Terms that must appear verbatim |

---

## 🐳 Docker Setup

```bash
# Start everything (app + Ollama)
docker-compose up -d

# Pull a model inside the Ollama container
docker exec ollama ollama pull mistral

# Access the app
open http://localhost:8501
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ using Streamlit, PyTorch, ChromaDB, and Ollama
</p>
