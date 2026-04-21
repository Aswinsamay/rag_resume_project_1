"""
utils.py
--------
Small helpers shared across the project:

- logging setup (so every module logs in a consistent format)
- a `timeit` context manager to measure latency (useful for the UI)
- project paths (data dir, chroma dir, embedding cache dir)

We keep this file deliberately small. Anything that is *truly* generic
and reusable lives here; everything else lives with the module that owns it.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# We anchor all paths on this file's location so the app works no matter where
# the user runs `streamlit run app.py` from.
ROOT_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT_DIR / "data"
CHROMA_DIR: Path = ROOT_DIR / "chroma_db"
FAISS_DIR: Path = ROOT_DIR / "faiss_index"
CACHE_DIR: Path = ROOT_DIR / ".embedding_cache"

for _p in (DATA_DIR, CHROMA_DIR, FAISS_DIR, CACHE_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a consistent format.

    Using a single factory avoids every module re-configuring logging
    (which can cause duplicate log lines).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
@contextmanager
def timeit(label: str = "") -> Iterator[dict]:
    """Context manager that measures elapsed wall-clock time.

    Usage::

        with timeit("retrieval") as t:
            ...
        print(t["elapsed_s"])

    We expose the timing via a dict so callers can read it AFTER the
    `with` block (unlike a plain variable, dict contents survive).
    """
    result: dict = {"label": label, "elapsed_s": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_s"] = time.perf_counter() - start
