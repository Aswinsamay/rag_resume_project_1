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
                             Toggle: "Compare RAG vs No-RAG".
        2. Chunk Preview   – see how the current strategy splits your PDF.
        3. Bad-Chunk Demo  – the headline teaching moment: same question,
                             run against BAD chunks and GOOD chunks, side by
                             side, so students *see* chunking matter.

Run locally (make sure Ollama is already running):
    ollama pull llama3.1:8b           # or mistral:7b
    ollama pull nomic-embed-text
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import streamlit as st
from langchain_core.documents import Document

from chunking import CHUNKING_STRATEGIES, chunk_documents
from embeddings import DEFAULT_EMBED_MODEL, get_embeddings
from generator import DEFAULT_LLM_MODEL, answer_with_rag, answer_without_rag
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
        "chat_history": [],        # List[dict] — {question, rag, no_rag, citations}
        "ingest_stats": None,      # dict with page/chunk counts + timing
        "bad_store": None,         # dedicated store for the "bad" demo
        "good_store": None,        # dedicated store for the "good" demo
        "demo_ready": False,
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
            "sentence: sentence-aware. bad: deliberately broken (for demo)."
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
    rebuild_demo_btn = st.button("Rebuild Bad-Chunk Demo", use_container_width=True)


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


def _build_demo_stores(paths: List[Path]) -> None:
    """Build TWO stores from the same PDFs: one with 'bad' chunks, one with
    'recursive' chunks. The demo asks the same question against both to show
    how chunking affects retrieval/answer quality."""
    pages = load_pdfs(paths)
    embeddings = get_embeddings(model=embed_model)

    bad_chunks = chunk_documents(pages, strategy="bad")
    good_chunks = chunk_documents(
        pages, strategy="recursive", chunk_size=800, chunk_overlap=120
    )

    # Persisting BOTH into Chroma would collide on disk, so we use FAISS for
    # the demo stores — they're in-memory and disposable.
    st.session_state.bad_store = build_vector_store(
        bad_chunks, embeddings, backend="faiss", reset=True
    )
    st.session_state.good_store = build_vector_store(
        good_chunks, embeddings, backend="faiss", reset=True
    )
    st.session_state.demo_ready = True


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

if rebuild_demo_btn:
    if not uploaded_files:
        st.sidebar.warning("Upload a PDF first to build the demo.")
    else:
        with st.spinner("Building demo stores (bad + good chunks)..."):
            saved = _save_uploads(uploaded_files)
            _build_demo_stores(saved)
        st.sidebar.success("Bad-chunk demo is ready. See the 'Bad-Chunk Demo' tab.")

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
tab_chat, tab_preview, tab_demo = st.tabs(
    ["Chat", "Chunk Preview", "Bad-Chunk Demo"]
)


# -------------------------- TAB 1: CHAT ------------------------------------
with tab_chat:
    st.subheader("Ask a question")

    compare_mode = st.toggle(
        "Compare RAG vs No-RAG side-by-side",
        value=True,
        help="Runs the question twice: once with retrieved context, once without.",
    )
    use_rag = st.toggle(
        "Use RAG (when comparison mode is OFF)",
        value=True,
        help="Only matters if the compare toggle above is off.",
    )

    question = st.chat_input("Ask something about your documents...")

    if question:
        if st.session_state.store is None:
            st.error("No vector store yet. Upload a PDF and click 'Ingest Documents'.")
        else:
            with timeit("retrieve") as t_ret:
                hits = retrieve(st.session_state.store, question, k=top_k)

            rag_answer = None
            no_rag_answer = None

            if compare_mode or use_rag:
                with timeit("generate_rag") as t_rag:
                    rag_answer = answer_with_rag(question, hits, model=llm_model)
            if compare_mode or not use_rag:
                with timeit("generate_no_rag") as t_no:
                    no_rag_answer = answer_without_rag(question, model=llm_model)

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "rag": rag_answer,
                    "no_rag": no_rag_answer,
                    "hits": hits,
                    "timing": {
                        "retrieve_s": round(t_ret["elapsed_s"], 2),
                        "rag_s": round(t_rag["elapsed_s"], 2) if rag_answer else None,
                        "no_rag_s": round(t_no["elapsed_s"], 2) if no_rag_answer else None,
                    },
                }
            )

    # Render chat history (newest first).
    for turn in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(turn["question"])

        with st.chat_message("assistant"):
            if turn["rag"] and turn["no_rag"]:
                col_rag, col_no = st.columns(2)
                with col_rag:
                    st.markdown("**With RAG (grounded)**")
                    st.write(turn["rag"])
                    st.caption(f"LLM time: {turn['timing']['rag_s']}s")
                with col_no:
                    st.markdown("**No RAG (LLM only)**")
                    st.write(turn["no_rag"])
                    st.caption(f"LLM time: {turn['timing']['no_rag_s']}s")
            elif turn["rag"]:
                st.markdown("**Answer (with RAG)**")
                st.write(turn["rag"])
                st.caption(f"LLM time: {turn['timing']['rag_s']}s")
            else:
                st.markdown("**Answer (no RAG)**")
                st.write(turn["no_rag"])
                st.caption(f"LLM time: {turn['timing']['no_rag_s']}s")

            # Retrieved chunks — displayed for transparency even in no-RAG mode.
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


# -------------------------- TAB 3: BAD-CHUNK DEMO --------------------------
with tab_demo:
    st.subheader("Why chunking matters")
    st.markdown(
        """
Click **Rebuild Bad-Chunk Demo** in the sidebar (after uploading a PDF).
It builds TWO vector stores from the same document:

- A **bad** store using 50-character, word-breaking chunks.
- A **good** store using the `recursive` splitter with ~800-char chunks.

Then ask the same question below against both. You'll typically see the
"bad" side retrieve fragments like `"expe"` + `"rience w"` + `"ith Py"`, and
the LLM either hallucinates or refuses. The "good" side retrieves complete
sentences and answers correctly.
"""
    )

    if not st.session_state.demo_ready:
        st.info("Demo not built yet. Upload a PDF and click 'Rebuild Bad-Chunk Demo'.")
    else:
        demo_q = st.text_input(
            "Demo question",
            value="What are the candidate's main technical skills?",
            key="demo_q",
        )
        if st.button("Run demo", type="primary"):
            with st.spinner("Running both sides..."):
                bad_hits = retrieve(st.session_state.bad_store, demo_q, k=top_k)
                good_hits = retrieve(st.session_state.good_store, demo_q, k=top_k)
                bad_ans = answer_with_rag(demo_q, bad_hits, model=llm_model)
                good_ans = answer_with_rag(demo_q, good_hits, model=llm_model)

            col_bad, col_good = st.columns(2)
            with col_bad:
                st.markdown("### BAD chunking (50-char, no boundaries)")
                st.error(bad_ans)
                with st.expander("Retrieved fragments"):
                    for i, h in enumerate(bad_hits, start=1):
                        st.text(f"[{i}] {h.citation}  |  dist={h.score:.3f}")
                        st.code(h.text)
            with col_good:
                st.markdown("### GOOD chunking (recursive, ~800 chars)")
                st.success(good_ans)
                with st.expander("Retrieved chunks"):
                    for i, h in enumerate(good_hits, start=1):
                        st.text(f"[{i}] {h.citation}  |  dist={h.score:.3f}")
                        st.code(h.text[:600] + ("..." if len(h.text) > 600 else ""))
