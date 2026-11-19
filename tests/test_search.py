"""Unit tests for retrieval search, with a fake embedder and a fake store (no Ollama or ChromaDB)."""

import pytest

from local_rag_pipeline.retrieval.search import search
from local_rag_pipeline.vector_store import Hit


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed_query(self, text):
        self.calls.append(text)
        return [1.0, 0.0, 0.0]


class FakeStore:
    def __init__(self, hits):
        self.hits = hits
        self.asked = None

    def search(self, vector, top_k=4):
        self.asked = (vector, top_k)
        return self.hits[:top_k]


HITS = [Hit(id=f"g.pdf:p{i}:c0", text=f"chunk {i}", source="g.pdf", page=i, distance=0.1 * i) for i in range(1, 7)]


def test_returns_top_k_hits_in_store_order():
    result = search("static IP?", FakeEmbedder(), FakeStore(HITS), top_k=3)
    assert [h.page for h in result.hits] == [1, 2, 3]


def test_question_is_embedded_after_warm_up():
    embedder = FakeEmbedder()
    search("  static IP?  ", embedder, FakeStore(HITS))
    assert embedder.calls == ["warm up", "static IP?"]


def test_store_receives_the_question_vector_and_top_k():
    store = FakeStore(HITS)
    search("q", FakeEmbedder(), store, top_k=5)
    assert store.asked == ([1.0, 0.0, 0.0], 5)


def test_timings_are_recorded():
    result = search("q", FakeEmbedder(), FakeStore(HITS))
    assert result.embed_ms >= 0 and result.search_ms >= 0


def test_empty_question_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        search("   ", FakeEmbedder(), FakeStore(HITS))