"""
retriever.py
------------
A thin wrapper around the vector store. It:

1. Runs similarity search.
2. Formats the retrieved chunks into a single `context` string that we can
   paste into the LLM prompt.
3. Returns a parallel list of `citation` dicts we can display in the UI.

We keep retrieval logic OUT of generator.py so we can display retrieved
chunks in the UI even when the user has RAG toggled off (for comparison).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from utils import get_logger
from vector_store import similarity_search

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """One retrieval hit, flattened into a UI-friendly shape."""
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    score: float  # distance from query (lower = more similar)

    @property
    def citation(self) -> str:
        return f"{self.source_file} (page {self.page_number})"


def retrieve(store: VectorStore, query: str, k: int = 4) -> List[RetrievedChunk]:
    """Return the top-k chunks as UI-friendly `RetrievedChunk` objects."""
    hits = similarity_search(store, query, k=k)
    results: List[RetrievedChunk] = []
    for doc, score in hits:
        md = doc.metadata or {}
        results.append(
            RetrievedChunk(
                text=doc.page_content,
                source_file=md.get("source_file", "unknown.pdf"),
                page_number=int(md.get("page_number", 0)),
                chunk_index=int(md.get("chunk_index", 0)),
                score=float(score),
            )
        )
    logger.info("Retrieved %d chunk(s) for query=%r", len(results), query)
    return results


def build_context(chunks: List[RetrievedChunk], max_chars: int = 6000) -> str:
    """Concatenate retrieved chunks into one `context` string for the prompt.

    Why include the citation INSIDE the context:
        The LLM can only cite what it sees. If we tag every chunk with
        `[source.pdf p.3]` the model can copy that tag into its answer.

    Why `max_chars`:
        Each model has a context window. 6000 chars is a safe default for
        llama3.1:8b / mistral:7b (both have ~8k token windows).
    """
    blocks: List[str] = []
    used = 0
    for i, ch in enumerate(chunks, start=1):
        header = f"[{i}] Source: {ch.source_file} | Page: {ch.page_number}"
        block = f"{header}\n{ch.text}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(blocks)
