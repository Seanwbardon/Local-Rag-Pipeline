"""Retrieval: find the chunks closest in meaning to a question.

This is the retrieval half of RAG on its own: no language model is involved. The question is
embedded with the same model used at ingestion, and the vector store returns the nearest
chunks with their file, page and cosine distance. The RAG chain will pass these chunks to
the language model as context.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from local_rag_pipeline.vector_store import Hit


@dataclass
class SearchResult:
    question: str
    hits: list[Hit]
    embed_ms: float    # time to turn the question into a vector
    search_ms: float   # time to find the nearest chunks


def search(question: str, embedder, store, top_k: int = 4, warm_up: bool = True) -> SearchResult:
    """Embed `question` and return the `top_k` nearest chunks, closest first.

    `embedder` needs `embed_query(text)`; `store` needs `search(vector, top_k)`.
    With `warm_up`, the embedding model is loaded first so its load time isn't
    counted in `embed_ms`.
    """
    question = question.strip()
    if not question:
        raise ValueError("The question is empty.")
    if warm_up:
        embedder.embed_query("warm up")

    t = time.perf_counter()
    vector = embedder.embed_query(question)
    embed_ms = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    hits = store.search(vector, top_k=top_k)
    search_ms = (time.perf_counter() - t) * 1000

    return SearchResult(question=question, hits=hits, embed_ms=embed_ms, search_ms=search_ms)