"""
app.py
------
Streamlit UI that ties everything together.

Layout:
    [ Sidebar ]
        - PDF uploader
        - Chunking strategy selector
        - Chunk size / overlap sliders
        - Vector backend (Chroma / FAISS)
        - Top-k slider
        - LLM + embedding model pickers
        - "Ingest documents" button

    [ Main ]
        Tabs:
        1. Chat            – ask questions, see retrieved chunks + answer.
        2. Chunk Preview   – see how the current strategy splits your PDF.

Run locally (make sure Ollama is already running):
    ollama pull llama3.1:8b           # or mistral:7b
    ollama pull nomic-embed-text
    streamlit run app.py
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import List

import streamlit as st
from langchain_core.documents import Document

from chunking import CHUNKING_STRATEGIES, chunk_documents
from embeddings import DEFAULT_EMBED_MODEL, get_embeddings
from generator import DEFAULT_LLM_MODEL, answer_with_rag
from ingestion import load_pdfs
from retriever import retrieve
from utils import DATA_DIR, get_logger, timeit
from vector_store import build_vector_store, load_vector_store

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Local RAG Playground",
    page_icon=":mag:",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "store": None,             # main vector store
        "chunks": [],              # List[Document] — last chunks produced
        "chat_history": [],        # List[dict] — {question, rag, hits, timing}
        "ingest_stats": None,      # dict with page/chunk counts + timing
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_init_state()


# ---------------------------------------------------------------------------
# Sidebar — controls + ingestion
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Settings")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Files are saved locally under ./data/",
    )

    st.subheader("Chunking")
    strategy = st.selectbox(
        "Strategy",
        CHUNKING_STRATEGIES,
        index=CHUNKING_STRATEGIES.index("recursive"),
        help=(
            "recursive: usually best. fixed_size: naive baseline. "
            "sentence: sentence-aware. bad: intentionally broken (50-char, no overlap)."
        ),
    )
    chunk_size = st.slider("Chunk size (chars)", 100, 2000, 800, step=50)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 500, 120, step=10)

    st.subheader("Retrieval")
    backend = st.selectbox("Vector store", ["chroma", "faiss"], index=0)
    top_k = st.slider("Top-k chunks to retrieve", 1, 10, 4)

    st.subheader("Models")
    llm_model = st.text_input("Ollama LLM model", value=DEFAULT_LLM_MODEL)
    embed_model = st.text_input("Ollama embedding model", value=DEFAULT_EMBED_MODEL)

    st.divider()
    ingest_btn = st.button("Ingest Documents", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_uploads(files) -> List[Path]:
    """Persist Streamlit upload buffers to ./data so we can re-ingest later."""
    paths: List[Path] = []
    for f in files:
        dest = DATA_DIR / f.name
        dest.write_bytes(f.getvalue())
        paths.append(dest)
    return paths


def _ingest(paths: List[Path]) -> None:
    """Full pipeline: load -> chunk -> embed -> store."""
    # Release any previously-held vector store BEFORE rebuilding. On Windows
    # an open Chroma client keeps `data_level0.bin` memory-mapped, which
    # blocks the directory reset. Dropping the reference + gc.collect()
    # lets the OS release those handles.
    st.session_state.store = None
    gc.collect()

    with timeit("load") as t_load:
        pages = load_pdfs(paths)
    with timeit("chunk") as t_chunk:
        chunks = chunk_documents(
            pages,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    with timeit("embed+index") as t_index:
        embeddings = get_embeddings(model=embed_model)
        store = build_vector_store(chunks, embeddings, backend=backend, reset=True)

    st.session_state.store = store
    st.session_state.chunks = chunks
    st.session_state.ingest_stats = {
        "files": [p.name for p in paths],
        "pages": len(pages),
        "chunks": len(chunks),
        "load_s": round(t_load["elapsed_s"], 2),
        "chunk_s": round(t_chunk["elapsed_s"], 2),
        "index_s": round(t_index["elapsed_s"], 2),
    }


# ---------------------------------------------------------------------------
# Handle sidebar actions
# ---------------------------------------------------------------------------
if ingest_btn:
    if not uploaded_files:
        st.sidebar.warning("Upload at least one PDF first.")
    else:
        with st.spinner("Ingesting..."):
            saved = _save_uploads(uploaded_files)
            _ingest(saved)
        st.sidebar.success("Ingest complete!")

# If the app was restarted but a Chroma index exists on disk, pick it back up.
if st.session_state.store is None:
    try:
        st.session_state.store = load_vector_store(
            get_embeddings(model=embed_model), backend=backend
        )
    except Exception as e:
        logger.debug("No existing store could be loaded: %s", e)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Local RAG Playground")
st.caption(
    "Upload PDFs -> chunk -> embed with Ollama -> retrieve top-k -> answer "
    "with a local LLM. 100% offline once models are pulled."
)

if st.session_state.ingest_stats:
    s = st.session_state.ingest_stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files", len(s["files"]))
    c2.metric("Pages", s["pages"])
    c3.metric("Chunks", s["chunks"])
    c4.metric("Index time (s)", s["index_s"])


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_chat, tab_preview = st.tabs(["Chat", "Chunk Preview"])


# -------------------------- TAB 1: CHAT ------------------------------------
with tab_chat:
    st.subheader("Ask a question")

    question = st.chat_input("Ask something about your documents...")

    if question:
        if st.session_state.store is None:
            st.error("No vector store yet. Upload a PDF and click 'Ingest Documents'.")
        else:
            with timeit("retrieve") as t_ret:
                hits = retrieve(st.session_state.store, question, k=top_k)

            with timeit("generate_rag") as t_rag:
                rag_answer = answer_with_rag(question, hits, model=llm_model)

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "rag": rag_answer,
                    "hits": hits,
                    "timing": {
                        "retrieve_s": round(t_ret["elapsed_s"], 2),
                        "rag_s": round(t_rag["elapsed_s"], 2),
                    },
                }
            )

    # Render chat history (newest first).
    for turn in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(turn["question"])

        with st.chat_message("assistant"):
            st.write(turn["rag"])
            st.caption(f"LLM time: {turn['timing']['rag_s']}s")

            with st.expander(
                f"Retrieved {len(turn['hits'])} chunk(s)  |  "
                f"retrieval: {turn['timing']['retrieve_s']}s"
            ):
                for i, h in enumerate(turn["hits"], start=1):
                    st.markdown(
                        f"**[{i}] {h.citation}**  "
                        f"<span style='color:gray'>distance={h.score:.4f}</span>",
                        unsafe_allow_html=True,
                    )
                    st.text(h.text[:800] + ("..." if len(h.text) > 800 else ""))
                    st.divider()


# -------------------------- TAB 2: CHUNK PREVIEW ---------------------------
with tab_preview:
    st.subheader("Preview chunks from the last ingest")
    st.caption(
        "Use this tab to *see* what your chunking settings actually produce. "
        "Good chunks are self-contained thoughts. Bad chunks are fragments."
    )
    if not st.session_state.chunks:
        st.info("Nothing to preview yet. Upload a PDF and click 'Ingest Documents'.")
    else:
        chunks: List[Document] = st.session_state.chunks
        st.write(f"Total chunks: **{len(chunks)}**  |  Strategy: **{strategy}**")

        # Summary stats — useful to eyeball whether chunk_size is reasonable.
        lengths = [len(c.page_content) for c in chunks]
        c1, c2, c3 = st.columns(3)
        c1.metric("Min length", min(lengths))
        c2.metric("Avg length", int(sum(lengths) / len(lengths)))
        c3.metric("Max length", max(lengths))

        n = st.slider("How many chunks to show?", 1, min(50, len(chunks)), 10)
        for i, c in enumerate(chunks[:n], start=1):
            md = c.metadata
            st.markdown(
                f"**Chunk {i}** — `{md.get('source_file')}` p.{md.get('page_number')} "
                f"(len={len(c.page_content)})"
            )
            st.text(c.page_content)
            st.divider()
