"""Unit tests for the ChromaDB vector store, using small hand-made vectors (no Ollama needed)."""

import pytest

pytest.importorskip("chromadb")

from local_rag_pipeline.ingest.splitter import Chunk  # noqa: E402
from local_rag_pipeline.vector_store import VectorStore, VectorStoreError  # noqa: E402


def chunk(i: int, page: int = 1) -> Chunk:
    return Chunk(id=f"guide.pdf:p{page}:c{i}", source="guide.pdf", page=page, chunk_index=i, text=f"chunk {i}")


# Three directions in 3D: each chunk points along one axis.
VECTORS = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "db", "test", embedding_model="all-minilm")


def test_add_and_count(store):
    store.add([chunk(0), chunk(1), chunk(2)], VECTORS)
    assert store.count() == 3


def test_adding_again_does_not_duplicate(store):
    store.add([chunk(0), chunk(1), chunk(2)], VECTORS)
    store.add([chunk(0), chunk(1), chunk(2)], VECTORS)
    assert store.count() == 3


def test_search_returns_closest_chunk_first_with_metadata(store):
    store.add([chunk(0), chunk(1, page=7), chunk(2)], VECTORS)
    hits = store.search([0.1, 0.9, 0.0], top_k=2)
    assert hits[0].id == "guide.pdf:p7:c1"
    assert (hits[0].source, hits[0].page) == ("guide.pdf", 7)
    assert hits[0].distance < hits[1].distance


def test_search_on_empty_store_returns_nothing(store):
    assert store.search([1.0, 0.0, 0.0]) == []


def test_reset_clears_everything(store):
    store.add([chunk(0)], VECTORS[:1])
    store.reset()
    assert store.count() == 0


def test_data_persists_on_disk(tmp_path):
    VectorStore(tmp_path / "db", "test", "all-minilm").add([chunk(0)], VECTORS[:1])
    assert VectorStore(tmp_path / "db", "test", "all-minilm").count() == 1


def test_different_embedding_model_is_refused(tmp_path):
    VectorStore(tmp_path / "db", "test", "all-minilm").add([chunk(0)], VECTORS[:1])
    with pytest.raises(VectorStoreError, match="built with 'all-minilm'"):
        VectorStore(tmp_path / "db", "test", "nomic-embed-text").count()


def test_mismatched_lengths_are_refused(store):
    with pytest.raises(VectorStoreError):
        store.add([chunk(0), chunk(1)], VECTORS[:1])