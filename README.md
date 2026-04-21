# Local RAG Playground

A 100% local Retrieval-Augmented Generation (RAG) app. Upload PDFs, ask questions, get grounded answers with page-level citations — all without sending data to any external API.

- **UI:** Streamlit
- **LLM + Embeddings:** [Ollama](https://ollama.com/) (runs locally, no API keys)
- **Vector store:** ChromaDB (default, persistent) or FAISS (optional, in-memory)
- **PDF loaders:** PyMuPDF (fast) with a pdfplumber fallback for tricky pages
- **Framework:** LangChain

## Features

- **PDF ingestion** with automatic fallback between PyMuPDF and pdfplumber, header/footer cleanup, and scanned-PDF detection.
- **Four chunking strategies** — `recursive`, `fixed_size`, `sentence`, and a deliberately broken `bad` mode — selectable from the UI.
- **Grounded answers** with a strict system prompt: the model must answer from retrieved context only, cite `[source.pdf, page N]` for every claim, and reply "I don't know" when the context doesn't cover the question.
- **RAG vs No-RAG comparison** — run the same question with and without retrieved context, side by side.
- **Bad-Chunk Demo** — builds two vector stores from the same PDF (one with broken chunks, one with good chunks) and answers the same question against both, to make the impact of chunking visible.
- **Chunk Preview tab** — inspect what your current chunking settings actually produce (min/avg/max length, raw text).
- **Embedding cache** on disk, so re-ingesting the same text is near-instant.
- **Latency metrics** shown on every ingest and every answer (load / chunk / index / retrieve / LLM times).

## Project layout

```
rag_app/
├── app.py            # Streamlit UI
├── ingestion.py      # PDF loading + cleaning (PyMuPDF + pdfplumber fallback)
├── chunking.py       # 4 chunking strategies
├── embeddings.py     # Ollama nomic-embed-text + disk cache
├── vector_store.py   # Chroma + FAISS behind a uniform interface
├── retriever.py      # top-k retrieval + citation formatting
├── generator.py      # Ollama LLM + RAG / no-RAG prompts
├── utils.py          # logging, paths, timing helpers
├── requirements.txt
└── data/             # uploaded PDFs (git-ignored)
```

## Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running — https://ollama.com/download

Pull the models once:

```bash
ollama pull llama3.1:8b           # or: ollama pull mistral:7b
ollama pull nomic-embed-text
```

On Windows the Ollama server starts automatically. On macOS/Linux, run `ollama serve` if it isn't already running.

## Setup

```bash
cd rag_app
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then in the browser:

1. Upload one or more PDFs from the sidebar.
2. Pick a chunking strategy (start with `recursive`) and adjust chunk size / overlap / top-k if you like.
3. Click **Ingest Documents**.
4. Use the **Chat** tab to ask questions. Toggle **Compare RAG vs No-RAG** to see both answers side by side.
5. Use the **Chunk Preview** tab to inspect how your strategy splits the PDF.
6. Use the **Bad-Chunk Demo** tab (after clicking **Rebuild Bad-Chunk Demo** in the sidebar) to compare answers from a poorly chunked store vs a well chunked store.

## Configuration

All settings are exposed in the sidebar — no code changes needed:

| Setting | Default | Notes |
|---|---|---|
| Chunking strategy | `recursive` | `recursive`, `fixed_size`, `sentence`, `bad` |
| Chunk size (chars) | 800 | |
| Chunk overlap (chars) | 120 | |
| Vector store | `chroma` | `chroma` (persistent) or `faiss` (in-memory) |
| Top-k | 4 | Number of chunks retrieved per question |
| LLM model | `llama3.1:8b` | Any Ollama chat model you have pulled |
| Embedding model | `nomic-embed-text` | Any Ollama embedding model you have pulled |

## Notes

- The first ingest is slow because Ollama has to embed every chunk. Subsequent ingests of the same text are near-instant thanks to the on-disk embedding cache at `rag_app/.embedding_cache/`.
- ChromaDB persists under `rag_app/chroma_db/`. Delete that folder to reset the index.
- If text extraction returns almost nothing, the ingestor logs a warning that the PDF is likely scanned / image-only. OCR (e.g. Tesseract) would be needed to handle those and is not included.
