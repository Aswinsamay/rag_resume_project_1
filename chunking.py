"""
chunking.py
-----------
Four chunking strategies, side-by-side, for teaching.

Why chunking matters (say this out loud in the video):
- Embeddings encode *one vector per chunk*. If the chunk is too big, the
  vector becomes a "blurry average" of many topics and retrieval gets worse.
- If the chunk is too small, a single idea is split across multiple chunks and
  the LLM never sees the full thought.
- If the chunk is split MID-SENTENCE or MID-WORD, meaning is destroyed.

Strategies implemented:
    1. fixed_size    – naive char-slice of a fixed length
    2. recursive     – LangChain's RecursiveCharacterTextSplitter (usually best)
    3. sentence      – split on sentence boundaries, with sentence-level overlap
    4. bad           – intentionally terrible: tiny, no overlap, cuts words.

Every strategy returns a `List[Document]`, each chunk preserving the parent
page's metadata AND adding `chunk_index` + `strategy` so we can debug later.
"""

from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import get_logger

logger = get_logger(__name__)

CHUNKING_STRATEGIES = ["recursive", "fixed_size", "sentence", "bad"]


# ---------------------------------------------------------------------------
# Strategy 1: Fixed-size character chunks
# ---------------------------------------------------------------------------
def _fixed_size_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Slide a fixed window across the text.

    Better than the 'bad' version because we use overlap, but still blind to
    sentence/paragraph boundaries. Useful as a baseline.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size - 1))
    step = chunk_size - overlap

    chunks: List[str] = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + chunk_size])
        i += step
    return chunks


# ---------------------------------------------------------------------------
# Strategy 3: Sentence-based chunks with sentence-level overlap
# ---------------------------------------------------------------------------
# Simple sentence splitter. Not perfect (abbreviations like "Dr." will fool it)
# but good enough for a tutorial — and it intentionally avoids pulling in nltk.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _sentence_chunks(text: str, target_chars: int, sentence_overlap: int) -> List[str]:
    """Group sentences into chunks of ~`target_chars` characters, with
    `sentence_overlap` sentences repeated between consecutive chunks.

    Sentence-level overlap is often a good default: you preserve context
    without duplicating huge amounts of text.
    """
    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        # Would adding this sentence push us over the target? Then flush.
        if current and current_len + len(sent) + 1 > target_chars:
            chunks.append(" ".join(current))
            # Carry the last N sentences forward as overlap.
            current = current[-sentence_overlap:] if sentence_overlap > 0 else []
            current_len = sum(len(s) + 1 for s in current)
        current.append(sent)
        current_len += len(sent) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


# ---------------------------------------------------------------------------
# Strategy 4: Intentionally BAD chunking (for the teaching demo)
# ---------------------------------------------------------------------------
def _bad_chunks(text: str) -> List[str]:
    """Split every N characters with NO overlap and NO word-boundary awareness.

    N is deliberately small (50 chars) so:
      - sentences get cut in half
      - words get cut in half
      - chunks carry almost no standalone meaning

    This is what we show first in the "Bad Chunk Demo" to prove why good
    chunking matters.
    """
    BAD_SIZE = 50
    return [text[i : i + BAD_SIZE] for i in range(0, len(text), BAD_SIZE)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def chunk_documents(
    docs: List[Document],
    strategy: str = "recursive",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Document]:
    """Split per-page Documents into smaller chunk-level Documents.

    The original page's metadata (source_file, page_number) is preserved on
    every chunk, and we add:
      - chunk_index: position within that source page
      - strategy:    which splitter produced it (useful for debugging)
    """
    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose one of {CHUNKING_STRATEGIES}."
        )

    logger.info(
        "Chunking %d page(s) with strategy=%s, size=%d, overlap=%d",
        len(docs), strategy, chunk_size, chunk_overlap,
    )

    # The recursive splitter is stateful-ish, so build it once.
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try the strongest separators first, fall back to weaker ones.
        # This is what makes "recursive" usually beat "fixed_size".
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )

    chunked: List[Document] = []
    for doc in docs:
        text = doc.page_content

        if strategy == "fixed_size":
            parts = _fixed_size_chunks(text, chunk_size, chunk_overlap)
        elif strategy == "recursive":
            parts = recursive_splitter.split_text(text)
        elif strategy == "sentence":
            parts = _sentence_chunks(text, target_chars=chunk_size, sentence_overlap=1)
        elif strategy == "bad":
            parts = _bad_chunks(text)
        else:  # pragma: no cover — guarded above
            parts = [text]

        for idx, part in enumerate(parts):
            if not part.strip():
                continue
            md = dict(doc.metadata)
            md["chunk_index"] = idx
            md["strategy"] = strategy
            chunked.append(Document(page_content=part, metadata=md))

    logger.info("Produced %d chunk(s).", len(chunked))
    return chunked
