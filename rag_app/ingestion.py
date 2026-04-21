"""
ingestion.py
------------
Load PDFs into a list of `Document` objects (LangChain's unit of data).

Design decisions:

1. **Two extractors, fast-path first.**
   - PyMuPDF (`fitz`) is ~10x faster than pdfplumber and handles most PDFs well.
   - pdfplumber is slower but better at tricky layouts (tables, columns).
   - We try PyMuPDF first; if a page returns no text, we fall back to pdfplumber
     for THAT page. This keeps the common case fast but still robust.

2. **Scanned PDFs (image-only) are detected, not OCR'd.**
   A scanned PDF yields almost no text from either library. We log a warning
   and skip it. Adding OCR would require Tesseract; out of scope for a tutorial.

3. **Cleaning.**
   We strip repeated headers/footers (same line appearing on most pages) and
   collapse whitespace. Clean input = better chunks = better retrieval.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import pdfplumber
from langchain_core.documents import Document

from utils import get_logger

logger = get_logger(__name__)

# If a page has fewer than this many non-whitespace chars after extraction,
# we treat it as "empty" and try the fallback extractor.
MIN_CHARS_PER_PAGE = 20


# ---------------------------------------------------------------------------
# Low-level extractors
# ---------------------------------------------------------------------------
def _extract_with_pymupdf(pdf_path: Path) -> List[str]:
    """Return a list of page texts (index = page number - 1)."""
    pages: List[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pages.append(page.get_text("text") or "")
    return pages


def _extract_page_with_pdfplumber(pdf_path: Path, page_index: int) -> str:
    """Extract a single page with pdfplumber. Used as a fallback."""
    with pdfplumber.open(pdf_path) as pdf:
        if page_index >= len(pdf.pages):
            return ""
        return pdf.pages[page_index].extract_text() or ""


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def _collapse_whitespace(text: str) -> str:
    """Replace runs of whitespace with a single space, preserving paragraphs."""
    # Keep paragraph breaks (double newlines), collapse everything else.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_repeated_headers_footers(pages: List[str]) -> List[str]:
    """Remove lines that appear on >60% of pages (likely headers/footers).

    We look at the first 2 and last 2 lines of each page and count how often
    each exact line appears across pages. Lines above the threshold get
    removed from every page.
    """
    if len(pages) < 3:
        # Not enough pages to reliably detect repeated boilerplate.
        return pages

    candidate_lines: Counter = Counter()
    for page in pages:
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        if not lines:
            continue
        # Top 2 and bottom 2 lines are typical header/footer positions.
        for ln in lines[:2] + lines[-2:]:
            candidate_lines[ln] += 1

    threshold = max(2, int(0.6 * len(pages)))
    boilerplate = {ln for ln, c in candidate_lines.items() if c >= threshold}

    if not boilerplate:
        return pages

    logger.info("Removing %d repeated header/footer line(s).", len(boilerplate))
    cleaned: List[str] = []
    for page in pages:
        kept = [ln for ln in page.splitlines() if ln.strip() not in boilerplate]
        cleaned.append("\n".join(kept))
    return cleaned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_pdf(pdf_path: str | Path) -> List[Document]:
    """Load one PDF into a list of per-page Documents.

    Each Document has:
      - page_content: cleaned text for that page
      - metadata: {"source_file": <filename>, "page_number": <1-indexed>}

    Per-page granularity is important for RAG: it lets us cite the exact page
    in answers ("see page 4 of resume.pdf").
    """
    pdf_path = Path(pdf_path)
    logger.info("Loading PDF: %s", pdf_path.name)

    # 1. Fast path: PyMuPDF for all pages.
    try:
        pages = _extract_with_pymupdf(pdf_path)
    except Exception as e:
        logger.warning("PyMuPDF failed on %s (%s). Falling back to pdfplumber.", pdf_path.name, e)
        pages = []

    # 2. Fallback per-page to pdfplumber for pages that came back empty.
    for i, page_text in enumerate(pages):
        if len(page_text.strip()) < MIN_CHARS_PER_PAGE:
            try:
                fallback = _extract_page_with_pdfplumber(pdf_path, i)
                if len(fallback.strip()) >= MIN_CHARS_PER_PAGE:
                    logger.info("Used pdfplumber fallback for page %d.", i + 1)
                    pages[i] = fallback
            except Exception as e:
                logger.warning("pdfplumber fallback failed for page %d: %s", i + 1, e)

    # 3. Scanned-PDF detection: if essentially NO page produced text, warn.
    total_chars = sum(len(p.strip()) for p in pages)
    if total_chars < MIN_CHARS_PER_PAGE * max(1, len(pages) // 4):
        logger.warning(
            "PDF '%s' appears to be scanned / image-only (very little text "
            "extracted). Consider OCR'ing it first (e.g. with Tesseract).",
            pdf_path.name,
        )

    # 4. Clean across all pages (header/footer removal needs global view).
    pages = _strip_repeated_headers_footers(pages)

    # 5. Build Document list.
    docs: List[Document] = []
    for i, page_text in enumerate(pages):
        cleaned = _collapse_whitespace(page_text)
        if not cleaned:
            continue
        docs.append(
            Document(
                page_content=cleaned,
                metadata={
                    "source_file": pdf_path.name,
                    "page_number": i + 1,  # 1-indexed is user-friendly
                },
            )
        )
    logger.info("Loaded %d non-empty page(s) from %s.", len(docs), pdf_path.name)
    return docs


def load_pdfs(pdf_paths: List[str | Path]) -> List[Document]:
    """Load multiple PDFs and concatenate their page-Documents."""
    all_docs: List[Document] = []
    for p in pdf_paths:
        all_docs.extend(load_pdf(p))
    return all_docs
