"""
vector_store.py
---------------
Two backends behind a tiny, uniform interface:

- Chroma (default): persistent on disk, easy to inspect, great for local dev.
- FAISS (optional): in-memory, extremely fast, great for benchmarks.

Both store:
    - chunk text        (what we'll pass to the LLM)
    - metadata          (source_file, page_number, chunk_index, strategy)
    - embedding vector  (for similarity search)

We expose three simple functions:
    - build_vector_store(docs, embeddings, backend, ...)  -> store
    - load_vector_store(embeddings, backend, ...)         -> store | None
    - similarity_search(store, query, k)                  -> List[(Document, score)]

Keeping both backends under the same shape means retriever.py doesn't care
which one is in use.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Literal, Optional, Tuple

from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from utils import CHROMA_DIR, FAISS_DIR, get_logger

logger = get_logger(__name__)

Backend = Literal["chroma", "faiss"]
DEFAULT_COLLECTION = "rag_docs"


# ---------------------------------------------------------------------------
# Build (from a batch of chunks)
# ---------------------------------------------------------------------------
def build_vector_store(
    docs: List[Document],
    embeddings: Embeddings,
    backend: Backend = "chroma",
    reset: bool = True,
    collection_name: str = DEFAULT_COLLECTION,
) -> VectorStore:
    """Create a fresh vector store from `docs` and persist it to disk.

    Why `reset=True` by default: during a tutorial we almost always want
    "ingest replaces everything" — otherwise old chunks from a previous PDF
    keep leaking into retrieval results and confuse the demo.
    """
    if not docs:
        raise ValueError("No documents to index. Upload & ingest a PDF first.")

    if backend == "chroma":
        if reset and CHROMA_DIR.exists():
            logger.info("Resetting Chroma dir at %s", CHROMA_DIR)
            shutil.rmtree(CHROMA_DIR)
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("Building Chroma index with %d chunk(s).", len(docs))
        store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=str(CHROMA_DIR),
        )
        return store

    if backend == "faiss":
        logger.info("Building FAISS index with %d chunk(s).", len(docs))
        store = FAISS.from_documents(documents=docs, embedding=embeddings)
        if reset and FAISS_DIR.exists():
            shutil.rmtree(FAISS_DIR)
        FAISS_DIR.mkdir(parents=True, exist_ok=True)
        store.save_local(str(FAISS_DIR))
        return store

    raise ValueError(f"Unknown backend: {backend}")


# ---------------------------------------------------------------------------
# Load (from disk, if an ingest has already happened)
# ---------------------------------------------------------------------------
def load_vector_store(
    embeddings: Embeddings,
    backend: Backend = "chroma",
    collection_name: str = DEFAULT_COLLECTION,
) -> Optional[VectorStore]:
    """Load a previously-persisted store. Returns None if nothing exists yet."""
    if backend == "chroma":
        # Chroma writes a sqlite file (`chroma.sqlite3`) on disk; use its
        # presence as a cheap "did an ingest happen?" check.
        if not (CHROMA_DIR / "chroma.sqlite3").exists():
            return None
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

    if backend == "faiss":
        if not (FAISS_DIR / "index.faiss").exists():
            return None
        # `allow_dangerous_deserialization=True` is required because FAISS
        # indexes are pickled. It's safe here because WE created the file.
        return FAISS.load_local(
            str(FAISS_DIR),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

    raise ValueError(f"Unknown backend: {backend}")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def similarity_search(
    store: VectorStore, query: str, k: int = 4
) -> List[Tuple[Document, float]]:
    """Return the top-k chunks most similar to `query`, with a distance score.

    Note on the score:
      - Chroma returns a *distance* (lower = more similar).
      - FAISS also returns a distance by default.
    We pass it through as-is and let the UI display it to the user.
    """
    # `similarity_search_with_score` is supported by both backends.
    return store.similarity_search_with_score(query, k=k)
