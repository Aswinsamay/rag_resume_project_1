# Local RAG Playground

A beginner-friendly, 100% local Retrieval-Augmented Generation demo built for a YouTube tutorial.

- **UI:** Streamlit
- **LLM + Embeddings:** [Ollama](https://ollama.com/) (no API keys, runs on your machine)
- **Vector store:** ChromaDB (default, persistent) or FAISS (optional)
- **PDF loaders:** PyMuPDF (fast) + pdfplumber (fallback)
- **Framework:** LangChain

The app lets you upload PDFs, pick a chunking strategy, toggle RAG on/off, and includes a **"Bad-Chunk Demo"** that shows side-by-side how bad chunking destroys answer quality — the most important teaching moment in the tutorial.

## Project layout

```
rag_app/
├── app.py            # Streamlit UI
├── ingestion.py      # PDF loading + cleaning (PyMuPDF + pdfplumber fallback)
├── chunking.py       # 4 chunking strategies incl. deliberately bad one
├── embeddings.py     # Ollama nomic-embed-text + disk cache
├── vector_store.py   # Chroma + FAISS behind a uniform interface
├── retriever.py      # top-k retrieval + citation formatting
├── generator.py      # Ollama LLM + RAG / no-RAG prompts
├── utils.py          # logging, paths, timing helpers
├── requirements.txt
└── data/             # your uploaded PDFs (git-ignored)
```

## Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running — https://ollama.com/download

Pull the models you want (one-time):

```bash
ollama pull llama3.1:8b           # or: ollama pull mistral:7b
ollama pull nomic-embed-text
```

Make sure the Ollama server is running in the background (on Windows it starts automatically after install; on Linux/macOS run `ollama serve` if needed).

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

Then in the UI:

1. **Upload** one or more PDFs in the sidebar.
2. Pick a **chunking strategy** (start with `recursive`).
3. Click **Ingest Documents**.
4. Ask questions in the **Chat** tab. Leave "Compare RAG vs No-RAG" on to see the difference.
5. Visit the **Chunk Preview** tab to see how your strategy splits the PDF.
6. Visit the **Bad-Chunk Demo** tab, click **Rebuild Bad-Chunk Demo** in the sidebar, then ask a question — watch the bad side fail and the good side succeed.

## Teaching flow suggestions (for the tutorial)

1. Ingest with `bad` strategy -> show failing retrieval -> "see, it's fragments."
2. Re-ingest with `recursive` -> show it working -> "one line change, huge difference."
3. Use the **Compare RAG vs No-RAG** toggle on a very specific question from the PDF (e.g. "What is the phone number on page 1?") — the No-RAG side will hallucinate or refuse, the RAG side will cite the exact page.
4. Use the **Bad-Chunk Demo** tab for the punchline: same PDF, same question, two answers.

## Notes

- The first ingest is slow because Ollama needs to embed every chunk. Subsequent ingests of the same text are near-instant thanks to the embedding cache in `.embedding_cache/`.
- ChromaDB persists under `rag_app/chroma_db/`. Deleting that folder resets the index.
- If extraction returns essentially no text, the ingestor logs a warning that the PDF is likely **scanned**; add OCR (Tesseract) if you need to handle those.
