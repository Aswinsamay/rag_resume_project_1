"""
generator.py
------------
LLM layer.

`answer_with_rag(question, chunks, ...)` builds a grounded prompt from the
retrieved context and asks the local Ollama model to answer it. The system
prompt forces the model to use only the provided passages, cite the source
file + page for every claim, and say "I don't know" when the context doesn't
cover the question.
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
# engineering" in this project. Its jobs are:
#   1) Force the model to use ONLY the provided context (no outside knowledge).
#   2) Force it to synthesize across ALL provided passages, not just one.
#   3) Encourage thorough, well-structured answers (the previous "be concise"
#      rule produced one-line replies that buried real detail).
#   4) Require inline citations like [source.pdf, page N] for every claim.
#   5) Make it say "I don't know" when the context truly doesn't cover the
#      question — the single biggest hallucination reducer.
RAG_SYSTEM_PROMPT = """You are a careful, thorough assistant answering \
questions about the user's documents.

You will be given several CONTEXT passages, each labelled with [N] and a \
source file + page number. Treat the CONTEXT as the only ground truth.

Rules you MUST follow:
1. Use ONLY information from the CONTEXT below. Do not add anything from \
   your own prior knowledge, even if you think it's correct.
2. Synthesize across ALL relevant passages. If multiple passages contribute \
   to the answer, combine them into one coherent response — do not anchor \
   on a single passage if others add detail.
3. Be thorough. Cover every relevant detail you find: conditions, exceptions, \
   sequencing, numbers, examples, special cases. Do not omit detail to be brief.
4. Structure the answer for readability:
   - Start with a one-sentence direct answer.
   - Then expand with bullet points or short paragraphs as appropriate.
5. Cite sources INLINE for every distinct claim, in the form \
   [source_file, page N]. Two adjacent claims from the same source can share \
   one citation at the end of the sentence.
6. If the answer is not contained in the CONTEXT, reply exactly:
   "I don't know based on the provided documents."

CONTEXT:
{context}
"""


# ---------------------------------------------------------------------------
# LLM factory (cached per-process so we don't re-create on every call)
# ---------------------------------------------------------------------------
_llm_cache: dict = {}


def get_llm(model: str = DEFAULT_LLM_MODEL, temperature: float = 0.1) -> ChatOllama:
    """Return a ChatOllama instance, cached by (model, temperature).

    Low temperature (0.1) because for RAG we want *faithful* reproduction of
    the context, not creative writing.

    `num_ctx=8192` is critical: Ollama's default context window is small
    (often 2048 tokens) and it silently truncates from the front. With 4-6
    retrieved chunks plus the system prompt + question + answer, that
    default isn't enough — chunks get dropped and answers feel shallow.
    8192 comfortably fits the prompt, all chunks, and a long answer.
    """
    key = (model, temperature)
    if key not in _llm_cache:
        logger.info("Creating ChatOllama model=%s temperature=%s num_ctx=8192", model, temperature)
        _llm_cache[key] = ChatOllama(
            model=model,
            temperature=temperature,
            num_ctx=8192,
        )
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


