# AUTOSAR HLD Document Analysis Assistant (RAG)

Tata Technologies TechPulse FY-26, Applied AI/ML, **Case Study CS1**.

Upload AUTOSAR High-Level Design (HLD) PDFs, ask questions in natural language, and get
answers that are **grounded in the documents with page-level citations**. If the documents
do not contain enough evidence, the assistant **refuses to answer** instead of guessing.

## Features
- Page-by-page PDF text extraction (pdfplumber), multiple PDFs at once
- Sentence-aware chunking with overlap; chunks never cross pages, so page numbers stay exact
- Local embeddings with `all-MiniLM-L6-v2` (no embedding API cost)
- FAISS (`IndexFlatIP` on normalised vectors = cosine similarity) for retrieval
- Google Gemini generates answers from the retrieved passages only
- Citations validated against what was retrieved, with expandable source excerpts
- Three-layer refusal logic (see below)
- Streamlit UI: example questions, chat history, metrics, transcript download, per-document removal

## Architecture

```
 PDF upload ─► pdfplumber (page by page) ─► clean text
                                              │
                                   sentence-aware chunking
                                   (page number kept per chunk)
                                              │
                              MiniLM embeddings (normalised)
                                              │
                                      FAISS IndexFlatIP
                                              │
 Question ─► embed ─► top-k search ─► similarity gate (MIN_SCORE)
                                              │ pass
                       Gemini (evidence-only prompt, JSON output)
                                              │
                 validate answerable flag + citation numbers
                                              │
                    Answer + [page citations]  or  Refusal
```

### Why it refuses (anti-hallucination design)
1. **Retrieval gate**: if no passage reaches the minimum cosine similarity, the LLM is never called.
2. **Model gate**: Gemini is told to answer only from numbered passages and to set `answerable=false` otherwise.
3. **Citation gate**: an answer with no valid citation to a retrieved passage is rejected.

Passages are also treated as untrusted data in the prompt, which mitigates prompt injection from document text.

## Project layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `rag_pipeline.py` | Extraction, chunking, embeddings, FAISS, Gemini, citations, refusal logic |
| `test_rag.py` | Offline pytest suite (fake embedder + fake LLM + generated PDFs) |
| `requirements.txt` | Dependencies |
| `.env.example` | Configuration template |
| `.gitignore` | Keeps secrets and caches out of Git |

## Installation

Requires **Python 3.11**.

```bash
# Windows (PowerShell)
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

```bash
# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY` (free key: https://aistudio.google.com/apikey).
You can also paste the key into the app sidebar instead.

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | none | Your Gemini key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Any Gemini model your key can access |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 150 | Characters per chunk / overlap |
| `TOP_K` | 5 | Passages retrieved per question |
| `MIN_SCORE` | 0.30 | Minimum cosine similarity to attempt an answer |

## Run

```bash
streamlit run app.py
```

Open http://localhost:8501, upload PDFs, click **Index documents**, then ask questions.
The first run downloads the embedding model (~90 MB).

## Test

```bash
pytest -v
```

Tests run offline and cover page tracking, chunking limits, FAISS retrieval, correct citations,
all refusal paths, malformed model output, corrupt/scanned PDFs, duplicate uploads and error handling.

## Suggested demo script
1. Upload an AUTOSAR HLD PDF and index it.
2. Ask a content question, such as the purpose of a module described in the document. Show the answer and the page citations.
3. Expand a citation and verify the excerpt on that page of the PDF.
4. Ask something unrelated ("What is the capital of France?") and show the refusal.
5. Ask about something AUTOSAR-related that your PDF does not cover to show the refusal when evidence is missing.
6. Raise `Minimum relevance` in the sidebar to show how strictness changes behaviour.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No extractable text found` | The PDF is scanned images. Run OCR first (e.g. OCRmyPDF). |
| `Gemini request failed` / 404 model | Set `GEMINI_MODEL` to a model available to your key. |
| `429` / quota errors | Wait and retry, or use a lighter model. |
| Assistant refuses too often | Lower `Minimum relevance` (try 0.20-0.25), or increase top-k. |
| Assistant answers weak matches | Raise `Minimum relevance` (0.35-0.45). |
| Slow first start | The embedding model is downloading; later runs use the cache. |
| `faiss` install fails | Use Python 3.11 and upgrade pip; install `faiss-cpu` only (not `faiss-gpu`). |
| `ModuleNotFoundError` | Activate the virtual environment and re-run `pip install -r requirements.txt`. |

## Limitations
- Text-only extraction: scanned PDFs need OCR, and content inside diagrams or images is not searched.
- Tables are extracted as plain text lines.
- Each question is answered independently (no multi-turn memory), which keeps answers traceable.
- The index lives in memory and is rebuilt when the app restarts.
- Always verify critical engineering decisions against the cited pages.