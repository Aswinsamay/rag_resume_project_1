"""
embeddings.py
-------------
Wraps Ollama's `nomic-embed-text` model behind a clean interface AND adds
a tiny on-disk cache so re-ingesting the same text doesn't re-embed.

=== What are embeddings? (read this out loud in the video) ===
An "embedding" turns a piece of text into a list of numbers (a vector)
such that *semantically similar* texts land close together in that vector
space. "dog" and "puppy" end up near each other; "dog" and "spreadsheet"
end up far apart.

=== What is cosine similarity? ===
Think of each embedding as an arrow from the origin. Cosine similarity
measures the ANGLE between two arrows (ignoring length):
    1.0 = same direction (very similar)
    0.0 = perpendicular  (unrelated)
   -1.0 = opposite       (very different)
RAG retrieval = "find the chunks whose arrows point most like my question."
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from utils import CACHE_DIR, get_logger

logger = get_logger(__name__)

DEFAULT_EMBED_MODEL = "nomic-embed-text"


# ---------------------------------------------------------------------------
# Cached wrapper
# ---------------------------------------------------------------------------
class CachedOllamaEmbeddings(Embeddings):
    """Ollama embeddings + a simple file-backed cache.

    Why: embeddings for the same chunk produce the same vector, so there's
    no point re-calling the model after the first run. This makes repeated
    ingests near-instant during development.

    Cache key = SHA1(model_name + '::' + text). Collisions are astronomically
    unlikely for our use case.
    """

    def __init__(self, model: str = DEFAULT_EMBED_MODEL, cache_dir: Path = CACHE_DIR) -> None:
        self.model = model
        self._inner = OllamaEmbeddings(model=model)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- internal helpers --------------------------------------------------
    def _cache_path(self, text: str) -> Path:
        h = hashlib.sha1(f"{self.model}::{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.json"

    def _read_cache(self, text: str) -> List[float] | None:
        p = self._cache_path(text)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            # Corrupt cache file — just ignore and re-embed.
            return None

    def _write_cache(self, text: str, vec: List[float]) -> None:
        try:
            self._cache_path(text).write_text(json.dumps(vec), encoding="utf-8")
        except Exception as e:
            logger.debug("Failed to write embedding cache: %s", e)

    # -- LangChain Embeddings interface -----------------------------------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of chunks; serve from cache where possible."""
        out: List[List[float]] = [None] * len(texts)  # type: ignore[list-item]
        missing_idx: List[int] = []
        missing_text: List[str] = []

        for i, t in enumerate(texts):
            cached = self._read_cache(t)
            if cached is not None:
                out[i] = cached
            else:
                missing_idx.append(i)
                missing_text.append(t)

        if missing_text:
            logger.info(
                "Embedding %d new chunk(s) via Ollama (%d served from cache).",
                len(missing_text), len(texts) - len(missing_text),
            )
            # One batched call to Ollama is much faster than N solo calls.
            fresh = self._inner.embed_documents(missing_text)
            for idx, vec, text in zip(missing_idx, fresh, missing_text):
                out[idx] = vec
                self._write_cache(text, vec)

        return out  # type: ignore[return-value]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single user query. We usually DON'T want to cache queries
        (they're unique per user turn), but the cost is trivial either way."""
        cached = self._read_cache(text)
        if cached is not None:
            return cached
        vec = self._inner.embed_query(text)
        self._write_cache(text, vec)
        return vec


def get_embeddings(model: str = DEFAULT_EMBED_MODEL) -> CachedOllamaEmbeddings:
    """Factory so the rest of the codebase never imports Ollama directly."""
    return CachedOllamaEmbeddings(model=model)
