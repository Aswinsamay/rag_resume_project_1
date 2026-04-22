"""
generator.py
------------
LLM layer. Two modes:

- `answer_with_rag(question, chunks, ...)`  -> grounded answer + citations
- `answer_without_rag(question, ...)`       -> direct LLM answer (baseline)

Both use the same Ollama model. The ONLY difference is whether we prepend
retrieved context and a "stay grounded" system prompt.

That's the whole point of the RAG-vs-no-RAG toggle in the UI: side by side,
students can see the LLM hallucinate on specific questions about their PDF
when context is absent, and answer correctly when context is present.
"""

from __future__ import annotations

from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from retriever import RetrievedChunk, build_context
from utils import get_logger

logger = get_logger(__name__)

DEFAULT_LLM_MODEL = "llama3.1:8b"  # swap for "mistral:7b" if you prefer


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
# The RAG system prompt is the single most important piece of "prompt
# engineering" in this project. Three jobs:
#   1) Tell the model it MUST use the provided context.
#   2) Tell the model to say "I don't know" if the context doesn't cover it
#      (this single line dramatically cuts hallucination).
#   3) Tell the model to cite sources using the [source.pdf p.N] pattern
#      that we already embedded into the context.
RAG_SYSTEM_PROMPT = """You are a careful assistant answering questions about \
the user's documents.

Rules you MUST follow:
1. Use ONLY the information in the CONTEXT below. Do not use outside knowledge.
2. If the answer is not contained in the CONTEXT, reply exactly:
   "I don't know based on the provided documents."
3. Cite the source(s) you used at the end of your answer in the form
   [source_file, page N]. Cite every claim.
4. Be concise. Prefer short, direct answers.

CONTEXT:
{context}
"""

NO_RAG_SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's \
question using your own knowledge. Be concise."""


# ---------------------------------------------------------------------------
# LLM factory (cached per-process so we don't re-create on every call)
# ---------------------------------------------------------------------------
_llm_cache: dict = {}


def get_llm(model: str = DEFAULT_LLM_MODEL, temperature: float = 0.1) -> ChatOllama:
    """Return a ChatOllama instance, cached by (model, temperature).

    Low temperature (0.1) because for RAG we want *faithful* reproduction of
    the context, not creative writing.
    """
    key = (model, temperature)
    if key not in _llm_cache:
        logger.info("Creating ChatOllama model=%s temperature=%s", model, temperature)
        _llm_cache[key] = ChatOllama(model=model, temperature=temperature)
    return _llm_cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def answer_with_rag(
    question: str,
    chunks: List[RetrievedChunk],
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.1,
) -> str:
    """Build the RAG prompt and return the LLM's answer text."""
    context = build_context(chunks)
    if not context.strip():
        # Defensive: if retrieval returned nothing, don't silently hallucinate.
        return "I don't know based on the provided documents."

    prompt = ChatPromptTemplate.from_messages(
        [("system", RAG_SYSTEM_PROMPT), ("human", "{question}")]
    )
    llm = get_llm(model=model, temperature=temperature)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content if hasattr(response, "content") else str(response)


def answer_without_rag(
    question: str,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.1,
) -> str:
    """Direct LLM answer with no retrieval. Used for the side-by-side demo."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", NO_RAG_SYSTEM_PROMPT), ("human", "{question}")]
    )
    llm = get_llm(model=model, temperature=temperature)
    chain = prompt | llm
    response = chain.invoke({"question": question})
    return response.content if hasattr(response, "content") else str(response)
